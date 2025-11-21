((nil .
      ((eval . (let ((root (projectile-project-root))
                     (cmd "project/CNN/cnn_model.py --config project/CNN/config.json")
                     )
                  (setq compile-command (concat "uv run " root cmd))
                  (setq gud-pdb-command-name (concat "python -m pdb " root cmd))
                  (setq gptel-org-branching-context nil)))
        (python-pytest-confirm . t)))
  (python-mode .
               ((eval . (pyvenv-activate (projectile-project-root)))
                (helm-dash-common-docsets . ("Git" "Python 3" "PyTorch" "NumPy" "Pandas" "pytest" "OpenCV Python"))))
  (org-mode .
            ((helm-dash-common-docset . ("Git" "Org_Mode" "Emacs Lisp")))))
