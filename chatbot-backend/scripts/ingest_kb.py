"""Enrichit la base de connaissances Qdrant à partir de deux sources :

1. Les fiches produits publiques d'une boutique WooCommerce, lues par la Store API
   (/wp-json/wc/store/v1/products), qui ne demande aucune clé d'API.
   Seules les informations stables sont indexées (nom, catégories, référence,
   description, caractéristiques, lien). Le prix et le stock ne sont PAS indexés :
   ils changent souvent et sont lus en temps réel par les outils WooCommerce.
2. Des fichiers texte ou Markdown (politiques, FAQ, conditions de vente) placés
   dans un dossier, par exemple kb/ : un fichier par sujet.

Les identifiants des points sont déterministes : relancer le script met à jour
les documents au lieu de créer des doublons.

Usage (depuis chatbot-backend/, ou dans le conteneur) :
  python scripts/ingest_kb.py --store-url https://technotools.tn --dry-run
  python scripts/ingest_kb.py --store-url https://technotools.tn --kb-dir kb
  docker exec chatbot_backend python scripts/ingest_kb.py --store-url https://technotools.tn --kb-dir kb
"""

from __future__ import annotations

import argparse
import asyncio
import html
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MAX_CHUNK_CHARS = 1200
BATCH_SIZE = 50

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n\s*\n+")


def strip_html(raw: str | None) -> str:
    """Texte brut à partir du HTML WooCommerce (balises retirées, entités décodées)."""
    if not raw:
        return ""
    text = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/div)\s*/?>", "\n", raw)
    text = html.unescape(_TAG_RE.sub(" ", text))
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Découpe par paragraphes, en regroupant les paragraphes courts."""
    paragraphs = [p.strip() for p in _BLANKS_RE.split(text or "") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:  # paragraphe trop long : coupe franche
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def product_to_text(p: dict[str, Any]) -> str:
    """Fiche produit lisible par le modèle, sans prix ni stock."""
    lines = [f"Produit : {html.unescape(p.get('name') or '').strip()}"]
    cats = [html.unescape(c.get("name", "")) for c in p.get("categories") or [] if c.get("name")]
    if cats:
        lines.append("Catégories : " + ", ".join(cats))
    if p.get("sku"):
        lines.append(f"Référence (SKU) : {p['sku']}")
    short = strip_html(p.get("short_description"))
    if short:
        lines.append(f"Résumé : {short}")
    attrs = []
    for a in p.get("attributes") or []:
        terms = [t.get("name", "") for t in a.get("terms") or [] if t.get("name")]
        if a.get("name") and terms:
            attrs.append(f"{a['name']} : {', '.join(terms)}")
    if attrs:
        lines.append("Caractéristiques : " + " ; ".join(attrs))
    desc = strip_html(p.get("description"))
    if desc and desc != short:
        lines.append(f"Description : {desc}")
    if p.get("permalink"):
        lines.append(f"Lien : {p['permalink']}")
    return "\n\n".join(lines)


async def fetch_store_products(store_url: str, max_pages: int) -> List[dict[str, Any]]:
    import httpx

    base = store_url.rstrip("/") + "/wp-json/wc/store/v1/products"
    products: List[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True,
                                 headers={"User-Agent": "chatbot-kb-ingest/1.0"}) as client:
        for page in range(1, max_pages + 1):
            r = await client.get(base, params={"per_page": 100, "page": page})
            if r.status_code == 400 and page > 1:  # page au-delà de la dernière
                break
            r.raise_for_status()
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            products.extend(batch)
            total_pages = int(r.headers.get("X-WP-TotalPages", page))
            print(f"  page {page}/{total_pages} : {len(batch)} produits")
            if page >= total_pages:
                break
    return products


def product_documents(products: Iterable[dict[str, Any]], store_key: str):
    from app.services.qdrant_service import Document

    docs = []
    for p in products:
        pid = p.get("id")
        name = html.unescape(p.get("name") or "").strip()
        for i, chunk in enumerate(chunk_text(product_to_text(p))):
            if i > 0:  # chaque passage garde le nom du produit pour rester compréhensible seul
                chunk = f"Produit : {name} (suite)\n\n{chunk}"
            docs.append(Document(
                id=f"product_{store_key}_{pid}_{i}",
                text=chunk,
                metadata={"source": "product", "product_id": pid, "store": store_key, "lang": "fr",
                          "title": html.unescape(p.get("name") or ""), "url": p.get("permalink") or ""},
            ))
    return docs


def kb_documents(kb_dir: Path, store_key: str):
    from app.services.qdrant_service import Document

    docs = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(Document(
                id=f"kb_{store_key}_{path.stem}_{i}",
                text=chunk,
                metadata={"source": path.stem, "store": store_key, "lang": "fr"},
            ))
    return docs


async def main_async(args: argparse.Namespace) -> int:
    from app.config import settings

    collection = args.collection or settings.QDRANT_COLLECTION
    docs = []

    if args.store_url:
        print(f"Lecture des produits publics de {args.store_url} ...")
        products = await fetch_store_products(args.store_url, args.max_pages)
        print(f"{len(products)} produits lus")
        docs += product_documents(products, args.store_key)

    if args.kb_dir:
        kb_dir = Path(args.kb_dir)
        if not kb_dir.is_dir():
            print(f"Dossier introuvable : {kb_dir}")
            return 2
        kb = kb_documents(kb_dir, args.store_key)
        print(f"{len(kb)} passages lus dans {kb_dir}")
        docs += kb

    if not docs:
        print("Rien à indexer (utilisez --store-url et/ou --kb-dir).")
        return 1

    if args.dry_run:
        print(f"[dry-run] {len(docs)} passages seraient indexés dans la collection {collection}. Exemple :\n")
        print(docs[0].text[:800])
        return 0

    from app.services.qdrant_service import upsert_documents

    for start in range(0, len(docs), BATCH_SIZE):
        await upsert_documents(collection=collection, documents=docs[start:start + BATCH_SIZE])
        print(f"  {min(start + BATCH_SIZE, len(docs))}/{len(docs)} passages indexés")
    print(f"Terminé : {len(docs)} passages dans la collection {collection}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrichit la base Qdrant (produits publics + fichiers de connaissances).")
    parser.add_argument("--store-url", help="URL de la boutique WooCommerce, ex. https://technotools.tn")
    parser.add_argument("--store-key", default="technotools", help="Clé de la boutique (identifiants des points)")
    parser.add_argument("--kb-dir", help="Dossier de fichiers .md/.txt (politiques, FAQ)")
    parser.add_argument("--collection", help="Collection Qdrant (défaut : QDRANT_COLLECTION)")
    parser.add_argument("--max-pages", type=int, default=20, help="Pages de 100 produits au maximum")
    parser.add_argument("--dry-run", action="store_true", help="Affiche ce qui serait indexé, sans écrire")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())