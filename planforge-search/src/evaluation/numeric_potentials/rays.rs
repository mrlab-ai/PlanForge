use std::fs;
use std::path::Path;

use planforge_cplex::{Constraint, Model, ObjectiveSense, SolveStatus};
use planforge_sas::state_registry::ConcreteStateView;
use serde_json::json;
use tracing::info;

use super::optimizer::PotentialSystem;
use super::{NumericPotentialFunction, NumericPotentialOptimizer, PotentialTask};

const RATIONAL_SCALE: i64 = 1_000_000;

pub(crate) struct Ray {
    function: NumericPotentialFunction,
    coefficients: Vec<f64>,
}

impl Ray {
    pub(crate) fn value(
        &self,
        state: ConcreteStateView<'_>,
        task: &PotentialTask,
        prop_scratch: &mut Vec<usize>,
        numeric_scratch: &mut Vec<f64>,
    ) -> Result<f64, String> {
        self.function
            .value(state, task, prop_scratch, numeric_scratch)
    }

    fn signature(&self, numeric_feature_count: usize) -> Vec<f64> {
        self.function.evaluation_signature(numeric_feature_count)
    }

    #[cfg(test)]
    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

pub(crate) struct RayGenerator {
    system: PotentialSystem,
    model: Model,
    epsilon: f64,
}

impl RayGenerator {
    pub(crate) fn new(optimizer: &NumericPotentialOptimizer, epsilon: f64) -> Result<Self, String> {
        let system = optimizer.homogeneous_system()?;
        let mut model = Model::new("numeric_potential_ray").map_err(|error| error.to_string())?;
        model
            .load(
                ObjectiveSense::Maximize,
                &system.variables,
                &system.constraints,
            )
            .map_err(|error| error.to_string())?;
        Ok(Self {
            system,
            model,
            epsilon,
        })
    }

    pub(crate) fn try_certify(
        &mut self,
        optimizer: &mut NumericPotentialOptimizer,
        state: ConcreteStateView<'_>,
    ) -> Result<Option<Ray>, String> {
        let objective = optimizer.objective_for_state(state, &self.system)?;
        self.model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        if self.model.solve().map_err(|error| error.to_string())? != SolveStatus::Optimal
            || self
                .model
                .objective_value()
                .map_err(|error| error.to_string())?
                <= self.epsilon
        {
            return Ok(None);
        }
        let primary = self
            .model
            .objective_value()
            .map_err(|error| error.to_string())?;
        let mut solution = self.model.solution().map_err(|error| error.to_string())?;

        let mut secondary = vec![0.0; self.system.variables.len()];
        let mut has_secondary = false;
        for &column in &self.system.weight_columns {
            let variable = self.system.variables[column];
            if variable.upper <= 0.0 && variable.lower < 0.0 {
                secondary[column] = 1.0;
                has_secondary = true;
            } else if variable.lower >= 0.0 && variable.upper > 0.0 {
                secondary[column] = -1.0;
                has_secondary = true;
            }
        }
        if has_secondary {
            let preserve = Constraint::new(
                primary - self.epsilon.max(1e-8),
                Model::infinity(),
                objective
                    .iter()
                    .enumerate()
                    .filter_map(|(column, coefficient)| {
                        (*coefficient != 0.0).then_some((column, *coefficient))
                    })
                    .collect(),
            );
            self.model
                .add_temporary_constraints(&[preserve])
                .map_err(|error| error.to_string())?;
            self.model
                .set_objective(&secondary)
                .map_err(|error| error.to_string())?;
            if self.model.solve().map_err(|error| error.to_string())? == SolveStatus::Optimal {
                solution = self.model.solution().map_err(|error| error.to_string())?;
            }
            self.model
                .clear_temporary_constraints()
                .map_err(|error| error.to_string())?;
        }
        self.certify_coefficients(optimizer, solution, &objective)
    }

    pub(crate) fn certify_native(
        &self,
        optimizer: &mut NumericPotentialOptimizer,
        coefficients: Vec<f64>,
        state: ConcreteStateView<'_>,
    ) -> Result<Option<Ray>, String> {
        let objective = optimizer.objective_for_state(state, &self.system)?;
        self.certify_coefficients(optimizer, coefficients, &objective)
    }

    fn certify_coefficients(
        &self,
        optimizer: &NumericPotentialOptimizer,
        mut coefficients: Vec<f64>,
        objective: &[f64],
    ) -> Result<Option<Ray>, String> {
        if coefficients.len() != self.system.variables.len() {
            return Ok(None);
        }
        let scale = coefficients
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        if !(scale > 0.0 && scale.is_finite()) {
            return Ok(None);
        }
        for coefficient in &mut coefficients {
            *coefficient /= scale;
            let Some(scaled) = scaled_integer(*coefficient) else {
                return Ok(None);
            };
            *coefficient = scaled as f64 / RATIONAL_SCALE as f64;
        }
        let objective_value = objective
            .iter()
            .zip(&coefficients)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        if objective_value <= self.epsilon
            || !verify_numerically(&self.system, &coefficients)
            || !verify_exact(&self.system, &coefficients, objective)
        {
            return Ok(None);
        }
        Ok(Some(Ray {
            function: optimizer.function_from_system_solution(&self.system, &coefficients),
            coefficients,
        }))
    }

    pub(crate) fn is_duplicate(
        &self,
        candidate: &Ray,
        existing: &[Ray],
        task: &PotentialTask,
    ) -> bool {
        let candidate = candidate.signature(task.features.len());
        let candidate_norm = norm(&candidate);
        if candidate_norm == 0.0 {
            return true;
        }
        existing.iter().any(|ray| {
            let signature = ray.signature(task.features.len());
            let denominator = candidate_norm * norm(&signature);
            denominator > 0.0 && dot(&candidate, &signature) / denominator > 0.99
        })
    }

    pub(crate) fn emit_certificate(
        &self,
        ray: &Ray,
        optimizer: &mut NumericPotentialOptimizer,
        state: ConcreteStateView<'_>,
        artifact_path: &Path,
    ) -> Result<bool, String> {
        let objective = optimizer.objective_for_state(state, &self.system)?;
        if !verify_exact(&self.system, &ray.coefficients, &objective) {
            return Ok(false);
        }
        let ray_values = scaled_values(&ray.coefficients)?;
        let objective_values = scaled_values(&objective)?;
        let variables = self
            .system
            .variables
            .iter()
            .enumerate()
            .map(|(column, variable)| {
                Ok(json!({
                    "name": format!("x{column}"),
                    "lower": scaled_bound(variable.lower)?,
                    "upper": scaled_bound(variable.upper)?,
                }))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let constraints =
            self.system
                .constraints
                .iter()
                .map(|row| {
                    let terms =
                        row.coefficients
                            .iter()
                            .map(|(column, coefficient)| {
                                Ok(json!([column, scaled_integer(*coefficient).ok_or_else(|| {
                            format!("ray row coefficient {coefficient} is not exactly scalable")
                        })?]))
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                    Ok(json!({
                        "lower": scaled_bound(row.lower)?,
                        "upper": scaled_bound(row.upper)?,
                        "terms": terms,
                    }))
                })
                .collect::<Result<Vec<_>, String>>()?;
        let fingerprint = fingerprint(&self.system, &objective);
        let artifact = json!({
            "fingerprint": fingerprint,
            "scale": RATIONAL_SCALE,
            "ray": ray_values,
            "variables": variables,
            "constraints": constraints,
            "objective": objective_values,
        });
        let encoded = serde_json::to_string_pretty(&artifact)
            .map_err(|error| format!("failed to encode numeric ray certificate: {error}"))?;
        fs::write(artifact_path, format!("{encoded}\n")).map_err(|error| {
            format!(
                "failed to write numeric ray certificate {}: {error}",
                artifact_path.display()
            )
        })?;
        let checker_path = format!("{}.checker.py", artifact_path.display());
        fs::write(&checker_path, CHECKER).map_err(|error| {
            format!("failed to write numeric ray checker {checker_path}: {error}")
        })?;
        info!(
            "Numeric ray certificate emitted: {}",
            artifact_path.display()
        );
        info!("Numeric ray certificate checker: {checker_path}");
        info!("Numeric ray exact checker outcome: passed");
        Ok(true)
    }
}

const CHECKER: &str = r#"#!/usr/bin/env python3
"""Exact checker for a dumped numeric-potential ray certificate."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "numeric_potential_ray_certificate.json"
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)

scale = int(data["scale"])
ray = [int(value) for value in data["ray"]]
assert len(ray) == len(data["variables"])
for value, variable in zip(ray, data["variables"]):
    if variable["lower"] is not None:
        assert value >= int(variable["lower"])
    if variable["upper"] is not None:
        assert value <= int(variable["upper"])
for row in data["constraints"]:
    activity = sum(int(coefficient) * ray[int(column)]
                   for column, coefficient in row["terms"])
    if row["lower"] is not None:
        assert activity >= int(row["lower"]) * scale
    if row["upper"] is not None:
        assert activity <= int(row["upper"]) * scale
objective = [int(value) for value in data["objective"]]
assert len(objective) == len(ray)
assert sum(coefficient * value
           for coefficient, value in zip(objective, ray)) > 0
assert data["fingerprint"]
print("CERTIFICATE VERIFIED", data["fingerprint"])
"#;

fn scaled_values(values: &[f64]) -> Result<Vec<i64>, String> {
    values
        .iter()
        .map(|value| {
            scaled_integer(*value)
                .ok_or_else(|| format!("ray certificate value {value} is not exactly scalable"))
        })
        .collect()
}

fn scaled_bound(value: f64) -> Result<Option<i64>, String> {
    if is_effectively_infinite(value) {
        Ok(None)
    } else {
        scaled_integer(value)
            .map(Some)
            .ok_or_else(|| format!("ray certificate bound {value} is not exactly scalable"))
    }
}

fn fingerprint(system: &PotentialSystem, objective: &[f64]) -> String {
    let mut hash = 1_469_598_103_934_665_603_u64;
    let mut add = |text: &str| {
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    };
    add(&format!(
        "{};{};",
        system.variables.len(),
        system.constraints.len()
    ));
    for (column, variable) in system.variables.iter().enumerate() {
        add(&format!(
            "x{column};{:?};{:?};",
            scaled_bound(variable.lower).ok(),
            scaled_bound(variable.upper).ok()
        ));
    }
    for row in &system.constraints {
        for (column, coefficient) in &row.coefficients {
            add(&format!("{column};{:?};", scaled_integer(*coefficient)));
        }
        add("|");
    }
    add("objective|");
    for coefficient in objective {
        add(&format!("{:?};", scaled_integer(*coefficient)));
    }
    format!("{hash:016x}")
}

fn scaled_integer(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let scaled = (value * RATIONAL_SCALE as f64).round();
    if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return None;
    }
    let recovered = scaled / RATIONAL_SCALE as f64;
    ((recovered - value).abs() <= 1e-10 * value.abs().max(1.0)).then_some(scaled as i64)
}

fn is_effectively_infinite(value: f64) -> bool {
    !value.is_finite() || value.abs() >= Model::infinity()
}

fn verify_numerically(system: &PotentialSystem, ray: &[f64]) -> bool {
    let tolerance = 1e-7;
    for (value, variable) in ray.iter().zip(&system.variables) {
        if (!is_effectively_infinite(variable.lower) && *value < variable.lower - tolerance)
            || (!is_effectively_infinite(variable.upper) && *value > variable.upper + tolerance)
        {
            return false;
        }
    }
    system.constraints.iter().all(|row| {
        let activity = row
            .coefficients
            .iter()
            .map(|(column, coefficient)| coefficient * ray[*column])
            .sum::<f64>();
        (is_effectively_infinite(row.lower) || activity >= row.lower - tolerance)
            && (is_effectively_infinite(row.upper) || activity <= row.upper + tolerance)
    })
}

fn verify_exact(system: &PotentialSystem, ray: &[f64], objective: &[f64]) -> bool {
    let Some(ray): Option<Vec<i64>> = ray.iter().map(|value| scaled_integer(*value)).collect()
    else {
        return false;
    };
    let Some(objective): Option<Vec<i64>> = objective
        .iter()
        .map(|value| scaled_integer(*value))
        .collect()
    else {
        return false;
    };
    for (value, variable) in ray.iter().zip(&system.variables) {
        if !is_effectively_infinite(variable.lower)
            && scaled_integer(variable.lower).is_none_or(|bound| *value < bound)
        {
            return false;
        }
        if !is_effectively_infinite(variable.upper)
            && scaled_integer(variable.upper).is_none_or(|bound| *value > bound)
        {
            return false;
        }
    }
    for row in &system.constraints {
        let mut activity = 0_i128;
        for &(column, coefficient) in &row.coefficients {
            let Some(coefficient) = scaled_integer(coefficient) else {
                return false;
            };
            activity += coefficient as i128 * ray[column] as i128;
        }
        if !is_effectively_infinite(row.lower)
            && scaled_integer(row.lower)
                .is_none_or(|bound| activity < bound as i128 * RATIONAL_SCALE as i128)
        {
            return false;
        }
        if !is_effectively_infinite(row.upper)
            && scaled_integer(row.upper)
                .is_none_or(|bound| activity > bound as i128 * RATIONAL_SCALE as i128)
        {
            return false;
        }
    }
    objective
        .iter()
        .zip(ray)
        .map(|(left, right)| *left as i128 * right as i128)
        .sum::<i128>()
        > 0
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}
