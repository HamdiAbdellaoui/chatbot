import pytest

from app.services.language_service import detect_language


@pytest.mark.parametrize(
    "text",
    [
        "Bonjour, je voudrais suivre ma commande, merci.",
        "Est-ce que la livraison est gratuite pour les commandes de plus de 100 dinars ?",
        "Le service client a été très réactif, merci infiniment.",
    ],
)
def test_detects_french(text):
    assert detect_language(text) == "fr"


@pytest.mark.parametrize(
    "text",
    [
        "السلام عليكم، أريد معرفة حالة طلبي.",
        "هل يمكنني استرجاع المنتج إذا لم يعجبني؟",
        "شكرا جزيلا على خدمتكم الممتازة.",
    ],
)
def test_detects_standard_arabic(text):
    assert detect_language(text) == "ar"


@pytest.mark.parametrize(
    "text",
    [
        "Salem, bech nsaal 3la commande mte3i, wislit walla?",
        "3andi problème fi payment, ma yekhdemch el carte.",
        "Yezzi nsibou el produit fi stock walla lazem nestanna?",
    ],
)
def test_detects_tunisian_darija(text):
    assert detect_language(text) == "darija"


def test_detects_code_switching_as_darija():
    text = "Merci brother, walla you sauvez ma journée!"
    assert detect_language(text) == "darija"


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!!???...", "🙂🙂🙂"])
def test_empty_or_unrecognizable_text_is_other(text):
    assert detect_language(text) == "other"


@pytest.mark.parametrize(
    "text",
    [
        "Je souhaite un devis pour Ottawa",
        "Livraison au 3eme étage avant 9h, colis de 5kg",
        "Une batterie 18V et un chargeur 5A",
        "Je cherche une lampe H7 et un PS5",
    ],
)
def test_french_without_false_darija_match(text):
    assert detect_language(text) == "fr"


@pytest.mark.parametrize(
    "text",
    [
        "Nahb nechri perceuse, 9adech soumha?",
        "Salam, 9adeh el prix?",
        "ahla, famma livraison l Sousse?",
        "Taw nji nchouf el magasin",
        "chnia el garantie?",
        "Salam, el commande ma3a livraison?",
    ],
)
def test_detects_more_darija(text):
    assert detect_language(text) == "darija"
