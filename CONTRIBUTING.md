# Contributing to DB Diagram to SQL Schema Converter

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to the DB Diagram to SQL Schema Converter.

## 🤝 How to Contribute

### Reporting Issues

If you encounter a bug or have a feature request:

1. **Search existing issues** to see if your issue has already been reported
2. **Use the issue templates** to provide all necessary information
3. **Include clear steps to reproduce** for bug reports
4. **Provide example images** when reporting schema extraction issues

### Submitting Code Changes

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/dbdiagram2sql.git
cd dbdiagram2sql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/test_models.py

# Run tests with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Format code with Black
black src/ tests/ examples/

# Check code style with flake8
flake8 src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Type checking with mypy
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## 📋 Coding Standards

### Python Code Style

- **PEP 8 compliance**: Follow Python's official style guide
- **Black formatting**: Use Black for code formatting (88 character limit)
- **Type hints**: All public functions must have type hints
- **Docstrings**: Use Google-style docstrings for all classes and functions

### Example Function

```python
def extract_schema(
    self, 
    image: Union[str, Path, Image.Image],
    max_new_tokens: int = 1024,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Extract database schema from an ER diagram image.
    
    Args:
        image: Path to image file or PIL Image object
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling for generation
        
    Returns:
        Dictionary containing the extracted schema information
        
    Raises:
        ValueError: If image format is not supported
        RuntimeError: If model inference fails
    """
    # Implementation here
    pass
```

### Testing Guidelines

- **Test coverage**: Aim for at least 80% test coverage
- **Unit tests**: Write unit tests for all utility functions
- **Integration tests**: Test complete workflows
- **Mock external dependencies**: Use mocks for Hugging Face models in tests

### Example Test

```python
def test_schema_extractor_initialization():
    """Test SchemaExtractor initialization with default parameters."""
    extractor = SchemaExtractor()
    
    assert extractor.model_id == "zodiac2525/Qwen2.5-VL-Diagrams2SQL"
    assert extractor.device in ["cuda", "cpu"]
    assert extractor.max_pixels == 1024 * 28 * 28
```

## 📚 Documentation

### Updating Documentation

- **README**: Update README.md for new features or changes
- **Docstrings**: Keep docstrings up to date with code changes
- **API docs**: Update API documentation for public interface changes
- **Examples**: Add examples for new features

### Documentation Style

- Use clear, concise language
- Include code examples where appropriate
- Use proper Markdown formatting
- Link to relevant sections and external resources

## 🔄 Pull Request Process

### Before Submitting

1. **Sync with upstream**: Rebase your branch on latest `main`
2. **Run tests**: Ensure all tests pass
3. **Check code quality**: Run linting and formatting tools
4. **Update docs**: Update relevant documentation
5. **Add changelog entry**: Document your changes

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Tested on GPU/CPU

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

### Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: Reviewer will test functionality
4. **Documentation**: Check for documentation completeness

## 🏗️ Project Structure

### Key Directories

```
src/
├── models/          # Model definitions and wrappers
├── data/           # Dataset and data processing
├── utils/          # Utility functions
└── *.py           # Main scripts (inference, training, etc.)

tests/              # Test suite
├── test_models.py  # Model tests
├── test_data.py    # Data processing tests
└── fixtures/       # Test data and fixtures

examples/           # Usage examples
docs/              # Documentation
configs/           # Configuration files
```

### Adding New Features

1. **Model improvements**: Add to `src/models/`
2. **Data processing**: Add to `src/data/`
3. **Utilities**: Add to `src/utils/`
4. **Scripts**: Add main scripts to `src/`
5. **Tests**: Add corresponding tests to `tests/`

## 🐛 Debugging Tips

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Model loading errors**: Check model ID and internet connection
3. **Import errors**: Ensure all dependencies are installed

### Debugging Tools

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python src/inference.py --image example.png

# Profile memory usage
python -m memory_profiler src/inference.py

# Run with pdb debugger
python -m pdb src/inference.py
```

## 📊 Performance Considerations

### Optimization Guidelines

- **Memory usage**: Be mindful of GPU memory constraints
- **Image processing**: Optimize image loading and preprocessing
- **Batch processing**: Implement efficient batching
- **Caching**: Cache model outputs when appropriate

### Benchmarking

```bash
# Run performance benchmarks
python src/benchmark.py --dataset test_data.json --max_samples 100

# Memory profiling
python -m memory_profiler examples/basic_usage.py
```

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create release notes
4. Tag release
5. Update model on Hugging Face Hub

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: dvnguyen02@email.com for direct contact

### Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## 🎯 Contribution Ideas

### Good First Issues

- Add support for new image formats
- Improve error handling
- Add more comprehensive tests
- Enhance documentation
- Create additional examples

### Advanced Contributions

- Optimize model inference speed
- Add new evaluation metrics
- Implement model quantization
- Add streaming inference support
- Create web API interface

### Dataset Contributions

- Add new schema domains
- Improve data annotation quality
- Create synthetic data generators
- Add multilingual support

## 🙏 Recognition

All contributors will be recognized in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to making database schema extraction more accessible and accurate!

---

**Questions?** Feel free to reach out via GitHub issues or email. We're here to help!
