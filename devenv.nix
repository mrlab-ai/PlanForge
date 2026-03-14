{ pkgs, lib, config, inputs, ... }:

{
  enterShell = ''
    export PATH="$PATH:${pkgs.vscode-extensions.vadimcn.vscode-lldb}/share/vscode/extensions/vadimcn.vscode-lldb/adapter"
  '';
  languages.rust.enable = true;
  languages.rust.channel = "stable";
  packages = [
    pkgs.vscode-extensions.vadimcn.vscode-lldb
    pkgs.taplo
  ];
}
