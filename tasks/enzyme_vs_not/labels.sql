-- Binary classification: enzyme (1) vs non-enzyme (0)
--
-- Definition (TO CONFIRM):
-- Enzyme = protein with at least one GO Molecular Function
--          under "catalytic activity" (GO:0003824)

SELECT DISTINCT
    a.code AS accession,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM protein_go_term_annotation pga
            JOIN go_terms gt
              ON pga.go_id = gt.go_id
            WHERE pga.protein_id = p.id
              AND gt.category = 'F'
              AND gt.go_id = 'GO:0003824'
        )
        THEN 1
        ELSE 0
    END AS label
FROM protein p
JOIN accession a
  ON a.protein_id = p.id
WHERE a.primary = TRUE;
