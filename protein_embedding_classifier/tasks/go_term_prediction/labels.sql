-- GO Term Prediction: multi-label
-- Each GO term becomes a binary label (0/1) for the protein

SELECT
    a.code AS accession,
    gt.go_id AS go_term
FROM protein p
JOIN protein_go_term_annotation pga
  ON pga.protein_id = p.id
JOIN go_terms gt
  ON gt.go_id = pga.go_id
JOIN accession a
  ON a.protein_id = p.id
WHERE a.primary = TRUE
ORDER BY a.code, gt.go_id;
