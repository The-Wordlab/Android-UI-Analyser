"""The Compose oracle uses observable leaves, without inventing pruned ancestry."""

from pathlib import Path

import pytest

from android_ui_analyser.assertions import evaluate_assertion_step
from android_ui_analyser.schema import Element
from android_ui_analyser.session_contracts import load_session_contract

CONTRACT = Path(__file__).parents[1] / "experiments/aua_controller/contracts/compose-sort.yaml"
PRODUCTS = {
    "alpine_mug": ("Alpine Mug", "$7.99"),
    "beacon_lamp": ("Beacon Lamp", "$12.50"),
    "cedar_notebook": ("Cedar Notebook", "$4.25"),
    "drift_puzzle": ("Drift Puzzle", "$18.00"),
}
ORDERS = {
    "name": list(PRODUCTS),
    "price": ["cedar_notebook", "alpine_mug", "beacon_lamp", "drift_puzzle"],
}


def leaf(rid: str, text: str, position: int = 0) -> Element:
    return Element(
        id=f"el:{rid}:{position}",
        type="TextView",
        resource_id=rid,
        text=text,
        bounds=(0, position * 20, 100, position * 20 + 10),
        center=(50, position * 20 + 5),
        parent=None,
        window="app",
    )


def screen(order: str) -> list[Element]:
    # Public fixture shape: labels and prices without collected card/grid ancestors.
    # No recorded device identities or raw logs are embedded in this synthetic screen.
    result = [leaf("compose_order_status", f"Current order: {order}")]
    for slug in ORDERS[order]:
        name, price = PRODUCTS[slug]
        result += [
            leaf(f"compose_name_{slug}", name, len(result)),
            leaf(f"compose_price_{slug}", price, len(result) + 1),
        ]
    return result


def passes(elements: list[Element], checkpoint: str) -> bool:
    contract = load_session_contract(file=CONTRACT)
    phase = (
        contract.cleanup
        if checkpoint == "cleanup"
        else next(item for item in contract.checkpoints if item.id == checkpoint)
    )
    assert phase is not None
    return all(evaluate_assertion_step(step, elements).ok for step in phase.assertions)


@pytest.mark.parametrize(
    "checkpoint,order",
    [
        ("default_name_order", "name"),
        ("price_sorted", "price"),
        ("name_order_restored", "name"),
    ],
)
def test_complete_pruned_compose_screen_passes(checkpoint: str, order: str) -> None:
    assert passes(screen(order), checkpoint)


@pytest.mark.parametrize(
    "order,checkpoint", [("name", "default_name_order"), ("price", "price_sorted")]
)
@pytest.mark.parametrize(
    "damage",
    [
        "wrong_price",
        "wrong_name",
        "swapped_prices",
        "wrong_product_order",
        "unpaired_traversal",
        "missing_price",
        "duplicate_name",
        "extra_product",
        "false_status",
    ],
)
def test_incomplete_or_mispaired_compose_screen_fails(
    order: str, checkpoint: str, damage: str
) -> None:
    elements = screen(order)
    if damage == "wrong_price":
        elements[2].text = "$0.00"
    elif damage == "wrong_name":
        elements[1].text = "Unexpected product"
    elif damage == "swapped_prices":
        elements[2].text, elements[4].text = elements[4].text, elements[2].text
    elif damage == "wrong_product_order":
        elements[1:5] = elements[3:5] + elements[1:3]
    elif damage == "unpaired_traversal":
        # Correct unique name/price text cannot compensate for mispaired traversal.
        elements[2], elements[4] = elements[4], elements[2]
    elif damage == "missing_price":
        elements.pop(2)
    elif damage == "duplicate_name":
        elements.append(elements[1].model_copy(update={"id": "el:duplicate"}))
    elif damage == "extra_product":
        elements += [leaf("compose_name_extra", "Extra"), leaf("compose_price_extra", "$1.00")]
    elif damage == "false_status":
        elements[0].text = "Current order: price" if order == "name" else "Current order: name"
    assert not passes(elements, checkpoint)


def test_home_cleanup_rejects_remaining_compose_content() -> None:
    home = [leaf("dev.aua.fixture:id/fixture_home", ""), leaf("open_compose_grid", "Compose grid")]
    assert passes(home, "cleanup")
    assert not passes(home + screen("name"), "cleanup")
    assert not passes(home + [leaf("compose_price_alpine_mug", "$7.99")], "cleanup")
    assert not passes(screen("name"), "cleanup")
