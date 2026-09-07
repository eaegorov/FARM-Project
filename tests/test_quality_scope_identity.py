"""CPU regression checks for the frozen semantic protocol, not model accuracy."""
import copy
import json
import unittest

from farm_runtime.quality.scope_identity import consensus, validate_response


def answer(**changes):
    row = dict(observation_a="Selected enclosure with its visible front and side.",
               observation_b="The corresponding enclosure viewed from another direction.",
               scope_a="one_independent_object", scope_b="one_independent_object",
               relation="same_independent_object", confidence="high",
               reason="Corresponding physical boundary and fixed fittings in the references.")
    row.update(changes)
    return row


def response(order, **changes):
    parsed = answer(**changes)
    return dict(scope_order=order, raw=json.dumps(parsed), parsed=parsed, validation_error=None)


class IdentityProtocolTest(unittest.TestCase):
    def test_symmetric_identity_accepts_opposite_orders_without_extent(self):
        self.assertTrue(consensus([response([11, 29]), response([29, 11])], [11, 29]))

    def test_explicit_neighbor_part_structural_uncertainty_veto(self):
        for key in ("scope_a", "scope_b"):
            for value in ("multiple_independent_objects", "only_part", "structural_surface", "unclear"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_response(json.dumps(answer(**{key: value})))

    def test_rejected_scope_can_be_recorded_as_negative(self):
        parsed = answer(scope_b="multiple_independent_objects", relation="part_or_collection")
        self.assertEqual(validate_response(json.dumps(parsed)), parsed)
        self.assertFalse(consensus([response([11, 29]), response([29, 11], **parsed)], [11, 29]))

    def test_low_confidence_or_identity_disagreement_never_accepts(self):
        for changes in ({"confidence": "medium"}, {"confidence": "low"},
                        {"relation": "different_objects"}, {"relation": "unclear"}):
            with self.subTest(changes=changes):
                self.assertFalse(consensus([response([11, 29]), response([29, 11], **changes)], [11, 29]))

    def test_two_calls_in_same_order_do_not_supply_consensus(self):
        self.assertFalse(consensus([response([11, 29]), response([11, 29])], [11, 29]))
        self.assertFalse(consensus([response([11, 29])], [11, 29]))

    def test_scope_binding_cannot_change_or_coerce_types(self):
        for order in ([11, 30], [11, 11], ["11", 29], [True, 29]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                consensus([response(order), response([29, 11])], [11, 29])

    def test_raw_output_cannot_be_reinterpreted(self):
        edited = response([29, 11], relation="different_objects")
        edited["parsed"] = answer()
        with self.assertRaises(ValueError):
            consensus([response([11, 29]), edited], [11, 29])

    def test_failed_validation_does_not_retry_or_accept(self):
        failed = response([29, 11])
        failed["validation_error"] = "model output invalid"
        self.assertFalse(consensus([response([11, 29]), failed], [11, 29]))

    def test_duplicate_negative_relation_cannot_be_overwritten_by_positive(self):
        raw = json.dumps(answer()).replace(
            '"relation": "same_independent_object"',
            '"relation": "different_objects", "relation": "same_independent_object"',
        )
        with self.assertRaisesRegex(ValueError, "duplicate identity response field"):
            validate_response(raw)
        overwritten = response([29, 11])
        overwritten["raw"] = raw
        with self.assertRaisesRegex(ValueError, "duplicate identity response field"):
            consensus([response([11, 29]), overwritten], [11, 29])

    def test_extent_fields_and_missing_observation_rejected(self):
        invalid = [answer(whole_scope="A"), answer(observation_a=" "), answer(scope_b=False)]
        missing = copy.deepcopy(answer())
        del missing["scope_b"]
        invalid.append(missing)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_response(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
