{ pkgs, lib, config, inputs, ... }:

{
  packages = [
    pkgs.marp-cli
    pkgs.azure-cli
    pkgs.azure-functions-core-tools
    pkgs.fzf
  ];

  dotenv.enable = true;
  dotenv.filename = [ ".env" ];

  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync = {
        enable = true;
      };
    };
  };

  # Jupyter & notebook scripts
  scripts.lab.exec = "uv run jupyter lab --ServerApp.token='totototo' --ServerApp.allow_remote_access=True";
  scripts.nb-txt.exec = "uv run jupytext --to py:percent notebook.ipynb";
  scripts.txt-nb.exec = "uv run jupytext --to notebook notebook.py";
  scripts.lab-sync.exec = "uv run jupytext --sync notebook.ipynb";

  # Marp presentation
  scripts.marp_html.exec = "marp presentation.md --html";
  scripts.marp_pdf.exec = "marp presentation.md --pdf";

  # Local app development
  scripts.app.exec = "uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000";
  scripts.func-start.exec = "cd app && func start";

  # Infrastructure scripts
  scripts.deploy.exec = "./script/deploy.sh";
  scripts.destroy.exec = "./script/destroy.sh";
  scripts.logs.exec = "./script/logs.sh";

  # Azure helpers
  scripts.az-login.exec = "az login";
  scripts.az-logs.exec = ''
    FUNC_NAME=$(awk '/^name:/ {print $2; exit}' infra/Pulumi.yaml)-func
    az webapp log tail --name "$FUNC_NAME" --resource-group "$(awk '/^name:/ {print $2; exit}' infra/Pulumi.yaml)"
  '';

  # Testing
  scripts.test.exec = "uv run pytest -v";
  scripts.lint.exec = "uv run ruff check . && uv run ruff format --check .";
  scripts.lint-fix.exec = "uv run ruff check --fix . && uv run ruff format .";
  scripts.typecheck.exec = "uv run pyright";

  env.PYTHONPATH = "${config.devenv.root}/.devenv/state/venv/lib/python3.12/site-packages:$PYTHONPATH";

  enterShell = ''
    ./script/ensure-project-name.sh 2>/dev/null || true
    export PATH="$PWD/.devenv/state/venv/bin:$PATH"
    echo "Commands: lab, marp_html, app, func-start, deploy, destroy, logs, test, lint"
  '';
}
