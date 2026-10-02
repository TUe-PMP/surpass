# SURPASS 1.0.0 publication checklist

The code and release artifacts are technically ready. The public repository is
`https://github.com/TUe-PMP/surpass`. Complete the following publication steps:

1. Push the reviewed source to the public GitHub repository.
2. Connect the repository to Zenodo before publishing the GitHub release.
3. Create the signed or annotated tag `v1.0.0` and publish the GitHub release.
4. Confirm that Zenodo archived the release and minted a DOI.
5. Add the version-specific Zenodo DOI to `CITATION.cff` and `README.md` on
   the continuing `main` branch.
6. Rebuild and recheck the wheel, source distribution, and full release ZIP if
   any file changes after the DOI is inserted.
7. Cite the version-specific DOI in the manuscript and Supplementary
   Information.

Do not overwrite the archived `v1.0.0` tag. Subsequent documentation-only
corrections should be released as `v1.0.1`.
