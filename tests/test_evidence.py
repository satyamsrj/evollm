import unittest
from evollm.core.evidence import EvidenceClass, EvidenceQualifier


class TestEvidenceQualifier(unittest.TestCase):
    def setUp(self):
        self.qualifier = EvidenceQualifier()

    def test_query_rejected(self):
        r = self.qualifier.qualify("What is the port number?")
        self.assertEqual(r.evidence_class, EvidenceClass.QUERY)
        self.assertFalse(r.accept_for_learning)
        self.assertIsNone(r.loss_target)

    def test_question_mark_rejected(self):
        r = self.qualifier.qualify("Is the server running?")
        self.assertFalse(r.accept_for_learning)

    def test_hypothetical_rejected(self):
        r = self.qualifier.qualify("Assume the drone uses Jetson Nano.")
        self.assertEqual(r.evidence_class, EvidenceClass.HYPOTHETICAL)
        self.assertFalse(r.accept_for_learning)
        self.assertEqual(r.rejection_reason, "HYPOTHETICAL_STATEMENT")

    def test_explicit_correction_accepted(self):
        r = self.qualifier.qualify("No, the port is 9800, not 9042.")
        self.assertEqual(r.evidence_class, EvidenceClass.EXPLICIT_CORRECTION)
        self.assertTrue(r.accept_for_learning)
        self.assertEqual(r.loss_target, "L_CORR")

    def test_correction_trigger_actually(self):
        r = self.qualifier.qualify("Actually, the motor model is 4006.")
        self.assertEqual(r.evidence_class, EvidenceClass.EXPLICIT_CORRECTION)
        self.assertTrue(r.accept_for_learning)

    def test_verified_fact_with_source(self):
        r = self.qualifier.qualify(
            "The official manual specifies motor model 4006.",
            source_reference="manual_v2.pdf#page=12"
        )
        self.assertEqual(r.evidence_class, EvidenceClass.VERIFIED_FACT)
        self.assertTrue(r.accept_for_learning)
        self.assertEqual(r.loss_target, "L_KNOW")
        self.assertTrue(r.provenance_verified)

    def test_unverified_assertion_rejected(self):
        r = self.qualifier.qualify("My drone uses Raspberry Pi 4.")
        self.assertEqual(r.evidence_class, EvidenceClass.VERIFIED_FACT)
        self.assertFalse(r.accept_for_learning)
        self.assertEqual(r.rejection_reason, "SOURCE_UNAVAILABLE")

    def test_preference_signal_disabled(self):
        r = self.qualifier.qualify("I prefer shorter answers.")
        self.assertEqual(r.evidence_class, EvidenceClass.PREFERENCE)
        self.assertFalse(r.accept_for_learning)
        self.assertEqual(r.rejection_reason, "L_PREF_DISABLED")


    def test_ordinary_user_text_ssl_when_allowed(self):
        r = self.qualifier.qualify("Today I walked to the park.", allow_ssl=True)
        self.assertEqual(r.evidence_class, EvidenceClass.ORDINARY_USER_TEXT)
        self.assertTrue(r.accept_for_learning)
        self.assertEqual(r.loss_target, "L_SSL")

    def test_ordinary_user_text_rejected_when_ssl_disallowed(self):
        r = self.qualifier.qualify("Today I walked to the park.", allow_ssl=False)
        self.assertFalse(r.accept_for_learning)
        self.assertEqual(r.rejection_reason, "NO_SIGNAL_DETECTED")


if __name__ == "__main__":
    unittest.main()

