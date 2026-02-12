-- Labels for subcellular localization task
SELECT DISTINCT
    a.code AS accession,
    CASE
        WHEN p.organelle = 'nucleus' THEN 0
        WHEN p.organelle = 'mitochondrion' THEN 1
        WHEN p.organelle = 'cytoplasm' THEN 2
        ELSE 3 -- other
    END AS label
FROM protein p
JOIN accession a
  ON a.protein_id = p.id
WHERE a.primary = TRUE;
