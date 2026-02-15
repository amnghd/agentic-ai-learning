# Week 1: Foundation & Environment Setup

## Project 1.1: Development Environment & Tooling

### Objectives
- Set up a professional development environment
- Configure containerization
- Establish CI/CD pipelines
- Create project structure following best practices

### Project Structure
```
week1/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── scripts/
│   ├── setup.sh
│   └── install_dependencies.sh
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

### Tasks

#### Task 1: Project Structure
Create a well-organized project structure:
- `src/` - Source code
- `tests/` - Test files
- `data/` - Data files (gitignored)
- `models/` - Model artifacts (gitignored)
- `notebooks/` - Jupyter notebooks
- `docs/` - Documentation
- `scripts/` - Utility scripts

#### Task 2: Docker Setup
- Create Dockerfile for development environment
- Set up docker-compose with services:
  - Main application container
  - PostgreSQL database
  - Redis cache
  - Jupyter notebook server

#### Task 3: Dependency Management
- Use Poetry or pip-tools for dependency management
- Pin all dependencies with versions
- Separate dev and production dependencies

#### Task 4: Pre-commit Hooks
- Black for code formatting
- Flake8/Pylint for linting
- MyPy for type checking
- Security checks (bandit)

#### Task 5: CI/CD Pipeline
- GitHub Actions workflow:
  - Run tests on PR
  - Code quality checks
  - Build Docker images
  - Security scanning

### Deliverables
1. Complete project structure
2. Working Docker environment
3. CI/CD pipeline
4. Documentation

### Evaluation Criteria
- [ ] Project structure follows best practices
- [ ] Docker environment runs successfully
- [ ] CI/CD pipeline passes all checks
- [ ] Code quality tools are configured
- [ ] Documentation is complete

### Resources
- [Python Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
