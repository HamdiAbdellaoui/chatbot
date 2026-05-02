import unittest

from app.services.pii_service import detect_pii, mask_pii


class TestPIIService(unittest.TestCase):
    def test_email_masking(self):
        text = "Contact me at john.doe+test@gmail.com please"
        masked = mask_pii(text)
        self.assertIn("[EMAIL]", masked)
        self.assertNotIn("john.doe+test@gmail.com", masked)

    def test_phone_international_masking(self):
        text = "Mon numéro: +216 22 333 444"
        masked = mask_pii(text)
        self.assertIn("[PHONE]", masked)

    def test_phone_bare_digits_masking(self):
        text = "22333444"
        masked = mask_pii(text)
        self.assertEqual(masked, "[PHONE]")

    def test_phone_bare_digits_order_id_guard(self):
        text = "order 22333444 please"
        masked = mask_pii(text)
        self.assertEqual(masked, text)

    def test_name_masking_french(self):
        text = "Je m'appelle Amine Ben Salah et je veux retourner ma commande"
        masked = mask_pii(text)
        self.assertIn("Je m'appelle [NAME]", masked)

    def test_name_masking_arabic(self):
        text = "اسمي محمد و نحب نرجّع الطلب"
        masked = mask_pii(text)
        self.assertIn("اسمي [NAME]", masked)

    def test_address_masking_french(self):
        text = "Livraison à 12 rue de la Republique Tunis"
        masked = mask_pii(text)
        self.assertIn("[ADDRESS]", masked)

    def test_multiple_pii_in_one_message(self):
        text = "Email a@b.com phone 22333444"
        masked = mask_pii(text)
        self.assertEqual(masked.count("[EMAIL]"), 1)
        self.assertEqual(masked.count("[PHONE]"), 1)

    def test_detect_pii_non_overlapping_sorted(self):
        text = "My name is Alice Doe, email alice@example.com, tel +216 22 333 444"
        matches = detect_pii(text)
        self.assertTrue(all(m.start < m.end for m in matches))
        self.assertEqual(matches, sorted(matches, key=lambda m: m.start))
        for i in range(1, len(matches)):
            self.assertGreaterEqual(matches[i].start, matches[i - 1].end)

    def test_detect_pii_includes_confidence_and_value(self):
        text = "Email: alice@example.com"
        matches = detect_pii(text)
        self.assertTrue(any(m.type == "EMAIL" for m in matches))
        email = next(m for m in matches if m.type == "EMAIL")
        self.assertIn("alice@example.com", email.value)
        self.assertGreaterEqual(email.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
