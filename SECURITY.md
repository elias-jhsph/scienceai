# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take security seriously at ScienceAI. If you discover a security vulnerability, please follow these steps:

### DO NOT

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** disclose the vulnerability publicly before it has been addressed

### DO

1. **Email us directly** at elias@eliastechlabs.com with:
   - A description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (optional)

2. **Use a descriptive subject line** like "Security Vulnerability in ScienceAI"

3. **Allow time for response** - We aim to respond within 48 hours

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
2. **Assessment**: We will investigate and assess the vulnerability
3. **Resolution**: We will work on a fix and coordinate disclosure timing with you
4. **Credit**: We will credit you in the release notes (unless you prefer to remain anonymous)

## Security Best Practices for Users

### API Key Management

- **Never commit API keys** to version control
- Use environment variables: `export OPENAI_API_KEY="your-key"`
- Or use the secure key storage in `~/Documents/ScienceAI/scienceai-keys.json`

### Data Privacy

- ScienceAI processes documents locally before sending to OpenAI
- Be aware that document content is sent to OpenAI's API for analysis
- Review OpenAI's data usage policies for your compliance needs

### Network Security

- ScienceAI runs a local web server on port 4242 by default
- The server binds to localhost only (127.0.0.1)
- Do not expose ScienceAI to public networks without additional security measures

## Known Security Considerations

1. **OpenAI API**: All AI-powered features require sending data to OpenAI's servers
2. **Local Storage**: Project data is stored unencrypted on the local filesystem
3. **PDF Processing**: PDFs are processed using PyMuPDF and Tesseract; ensure these dependencies are up to date

## Dependency Security

We use Dependabot to monitor dependencies for known vulnerabilities. Security updates are prioritized and released as patch versions.
