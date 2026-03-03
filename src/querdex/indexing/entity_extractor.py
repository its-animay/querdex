from __future__ import annotations

import re
from collections import defaultdict

from querdex.schemas import EntityRef, Section, TreeNode
from querdex.utils.tree_ops import walk_nodes

_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_XREF_RE = re.compile(r"\b(?:Table|Section|Appendix|Figure)\s+[A-Za-z0-9\.\-]+\b", re.IGNORECASE)


class EntityExtractor:
    """Extracts entities and cross-references from section content."""

    def pre_extract(self, sections: list[Section]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
        section_entities: dict[str, set[str]] = {}
        section_xrefs: dict[str, list[str]] = {}

        for section in sections:
            section_entities[section.section_id] = self._extract_entities(section.content)
            refs = [" ".join(xref.split()) for xref in _XREF_RE.findall(section.content)]
            section_xrefs[section.section_id] = refs

        return section_entities, section_xrefs

    def finalize(
        self,
        doc_id: str,
        sections: list[Section],
        tree_root: TreeNode,
        section_entities: dict[str, set[str]],
        section_xrefs: dict[str, list[str]],
    ) -> tuple[dict[str, list[EntityRef]], dict[str, list[str]]]:
        page_to_node = self._page_to_leaf_node(tree_root)
        entity_map: dict[str, list[EntityRef]] = defaultdict(list)
        cross_refs: dict[str, list[str]] = defaultdict(list)
        section_lookup = {s.section_id: s for s in sections}

        for section_id, entities in section_entities.items():
            section = section_lookup[section_id]
            node_id = page_to_node.get(section.page_number, "node_0000")
            refs = EntityRef(doc_id=doc_id, node_id=node_id)

            for value in entities:
                entity_map[value].append(refs)

            for normalized in section_xrefs.get(section_id, []):
                cross_refs[node_id].append(normalized)
                entity_map[normalized].append(refs)

        deduped_map = {entity: self._dedupe_refs(refs) for entity, refs in entity_map.items()}
        deduped_refs = {node_id: sorted(set(values)) for node_id, values in cross_refs.items()}
        return deduped_map, deduped_refs

    @staticmethod
    def _extract_entities(content: str) -> set[str]:
        entities: set[str] = set()
        entities.update(_CAPITALIZED_RE.findall(content))
        entities.update(_NUMBER_RE.findall(content))
        return {e.strip() for e in entities if e.strip()}

    @staticmethod
    def _page_to_leaf_node(tree_root: TreeNode) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for node in walk_nodes(tree_root):
            if node.children:
                continue
            for page in range(node.start_page, node.end_page + 1):
                mapping[page] = node.node_id
        return mapping

    @staticmethod
    def _dedupe_refs(refs: list[EntityRef]) -> list[EntityRef]:
        seen: set[tuple[str, str]] = set()
        deduped: list[EntityRef] = []
        for ref in refs:
            key = (ref.doc_id, ref.node_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped
