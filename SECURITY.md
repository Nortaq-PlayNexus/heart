# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private security advisory on GitHub:

1. Go to the **Security** tab
2. Click **Report a vulnerability**
3. Fill in the details

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Security Features

- Secret scanning: **enabled**
- Push protection: **enabled**
- Dependabot alerts: **enabled** (Settings ? Security ? Dependabot alerts)

## No Hardcoded Secrets

All tokens and API keys are loaded from environment variables. Never commit secrets.

## Local-First Architecture

All AI inference runs locally via Ollama/LM Studio. No data leaves your network.