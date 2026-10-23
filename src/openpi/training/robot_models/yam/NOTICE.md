# YAM FK-only model provenance

`yam_fk.xml` is a modified FK-only derivative of:

- Repository: https://github.com/amazon-far/abc
- Source file: `assets/put_bottles/assets/i2rt_yam/yam.xml`
- Commit: `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`
- Source SHA-256: `a797c4c27d1c074eb24b04d9cf13cef9a66589f135ebf35bb22f2be3b86f7d28`
- License: MIT License, copyright 2025 i2rt robotics

Modifications: removed mesh/material assets, all geoms, lights, finger
subtrees, equality constraints, actuators and keyframes. Retained the exact
six-axis body/joint chain, link inertials and `grasp_site` transform required
for standalone forward kinematics.

This copy is taken from `data_filter_pipeline/assets/robots/yam/yam_fk.xml`,
the FK model accompanying the ABC-130K consistency checks. Its SHA-256 is
`1aa7d23998fbb9ef09f7907c84b2e45d6195961c2fe5996a8d470459f654dbea`.
