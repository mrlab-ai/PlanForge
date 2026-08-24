{ pkgs, lib, config, inputs, ... }:

{
  enterShell = ''
    export PATH="$PATH:${pkgs.vscode-extensions.vadimcn.vscode-lldb}/share/vscode/extensions/vadimcn.vscode-lldb/adapter"
  '';
  languages.rust.enable = true;
  languages.rust.channel = "stable";
  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };
  packages = [
    pkgs.vscode-extensions.vadimcn.vscode-lldb
    pkgs.taplo
    pkgs.valgrind
    pkgs.kdePackages.kcachegrind
    pkgs.cargo-flamegraph
    pkgs.samply
  ];
}
