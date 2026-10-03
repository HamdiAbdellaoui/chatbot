# WooCommerce MCP server — prototype (POC)

## Statut : prototype, PAS le chemin de production

Ce dossier est une **preuve de concept** montrant comment les capacités WooCommerce
du chatbot pourraient être exposées via le **Model Context Protocol** au lieu du
function calling OpenAI.

En production, le chatbot n'utilise **pas** ce serveur : il passe par
`app/services/llm_service.py` (function calling OpenAI) → `app/services/woocommerce_service.py`.
Ce chemin est inchangé, et `infra/docker-compose.yml` ne déploie pas ce serveur.

Ce POC **réutilise** le code de production plutôt que de le dupliquer :

| Besoin | Réutilise |
|---|---|
| Appels REST WooCommerce | `app/services/woocommerce_service.py` (`WooCommerceClient`) |
| Credentials par store | `app/services/store_context_service.py` (mapping `STORES_JSON`) |

## Structure

| Fichier | Rôle |
|---|---|
| `woocommerce_mcp_server.py` | Serveur MCP : enregistre les 2 tools, transport stdio |
| `tools.py` | Implémentation des tools (sans dépendance au SDK `mcp`, donc testable) |
| `demo_client.py` | Client MCP minimal : liste les tools et en appelle un |
| `requirements.txt` | Dépendance `mcp` (isolée de l'image de production) |

### Tools exposés (2, volontairement)

- `search_products(query: str, store_id: str) -> list[dict]`
- `get_price_and_stock(product_id: int, store_id: str) -> dict`

Le `store_id` est la **clé de store** telle que définie dans `STORES_JSON`
(ex. `store_a`), pas un `inbox_id` Chatwoot.

## Installation

```powershell
# Depuis chatbot-backend/
pip install -r mcp_server/requirements.txt
```

## Lancer le serveur en standalone

```powershell
# Depuis chatbot-backend/ (important : le .env est résolu depuis le cwd)
python mcp_server/woocommerce_mcp_server.py
```

Le serveur parle MCP sur **stdin/stdout** : lancé seul, il attend simplement des
messages du protocole et n'affiche rien. C'est normal — utilisez l'inspecteur ou
le client de démo ci-dessous pour interagir avec lui.

## Tester — option A : client de démo (recommandé pour le rapport)

```powershell
# Depuis chatbot-backend/
python mcp_server/demo_client.py
```

Avec des paramètres explicites :

```powershell
python mcp_server/demo_client.py --query iphone --store-id store_a
```

Le script lance le serveur en sous-processus, fait le handshake MCP, affiche les
tools exposés avec leur schéma JSON, puis appelle `search_products` et affiche la
réponse brute — c'est cette sortie qui sert de preuve d'exécution.

## Tester — option B : inspecteur MCP officiel

```powershell
# Depuis chatbot-backend/
mcp dev mcp_server/woocommerce_mcp_server.py
```

Ouvre l'inspecteur web où les tools peuvent être appelés à la main. La commande
exacte dépend de la version du SDK ; si `mcp dev` n'existe pas dans votre
version, utilisez l'option A.

## Prérequis pour obtenir de vrais produits

Les tools renvoient de vraies données uniquement si un store avec des credentials
WooCommerce est configuré dans `STORES_JSON` (voir `.env.example`). Exemple :

```
STORES_JSON='{"store_a": {"match": {"inbox_ids": [1]}, "qdrant_collection": "store_a", "woocommerce": {"base_url": "https://store-a.tld", "consumer_key": "ck_...", "consumer_secret": "cs_..."}}}'
```

Sans cette config, le handshake MCP et le listing des tools fonctionnent quand
même (le protocole est démontrable), mais l'appel de tool renvoie une erreur
explicite du type `Store 'store_a' has no WooCommerce credentials configured` /
`Unknown store_id 'store_a'`.
