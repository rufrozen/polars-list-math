# Publishing

This project is published to PyPI from GitHub Actions. The release workflow
builds a source distribution and wheels for CPython 3.12 and 3.13 on Linux,
macOS, and Windows, then uploads them with PyPI Trusted Publishing.

## One-time PyPI setup

Create a trusted publisher for the project on PyPI:

| Field | Value |
| --- | --- |
| PyPI project | `polars-list-math` |
| Owner | `rufrozen` |
| Repository | `polars-list-math` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

For the first release, PyPI can create the project from a pending trusted
publisher. No API token is needed in GitHub.

## Release checklist

1. Update the version in `pyproject.toml` and `rust/Cargo.toml`.
2. Run the local checks:

   ```bash
   make lint
   make test
   make check-dist
   ```

3. Commit the release changes.
4. Push `main`.
5. Create and push a version tag:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

6. Watch the `Publish` workflow. It should upload the final files from the
   `dist-*` artifacts to PyPI.

## Local package check

`make check-dist` builds the local source distribution and wheel, then runs
`twine check` against every file in `dist/`.

For an upload dry run, use TestPyPI from a separate workflow or local `twine`
configuration. Do not add PyPI API tokens to this repository; Trusted
Publishing is the default release path.
