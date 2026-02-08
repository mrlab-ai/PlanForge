use crate::translate::sas::SASTask;
use crate::translate::sas_tasks;
use std::fs::File;

pub fn write_sas(task: &SASTask, path: &std::path::Path) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    let py_task = sas_tasks::from_internal(task);
    py_task.output(&mut f)?;
    Ok(())
}
