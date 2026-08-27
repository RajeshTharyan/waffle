# Security

The Streamlit UI accepts **local file uploads** (`.txt` and `.md`) and pasted
text. It does **not** fetch user-supplied URLs. Citation-like URL strings in
the text are counted as features; they are not retrieved.

## What the app does with uploads

- Files are read in the Streamlit session and decoded as UTF-8 (`errors="ignore"`).
- Content is scored in memory. Nothing is written to disk by the scoring path.
- There is no authentication, multi-tenant isolation, or upload size quota
  beyond whatever the Streamlit host enforces.

Treat any public deployment as **untrusted input**. Do not paste secrets. If
you deploy a copy, put it behind your own access control and keep Streamlit
secrets out of git (see `.gitignore` for `.streamlit/secrets.toml`).

The checked-in `.streamlit/config.toml` disables XSRF protection and CORS so
Codespaces / Docker previews can connect. That is convenient for a demo host,
not a hardening baseline.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Email the
repository owner via the address on their GitHub profile, or use GitHub's
private vulnerability reporting if it is enabled on this repository.
