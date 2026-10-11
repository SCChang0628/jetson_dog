"""
Inspect Jetson Dog USDA structure.

Python 3.12
No Isaac Sim runtime required.
"""

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

USD_PATH = (
    PROJECT_ROOT
    / "src"
    / "jetson_dog"
    / "assets"
    / "data"
    / "Dog_v1.usda"
)


EXPECTED_JOINTS = [
    "FL_shoulder_Joint",
    "RL_shoulder_Joint",
    "FR_shoulder_Joint",
    "RR_shoulder_Joint",
    "FL_thigh_Joint",
    "RL_thigh_Joint",
    "FR_thigh_Joint",
    "RR_thigh_Joint",
    "FL_calves_Joint",
    "RL_calves_Joint",
    "FR_calves_Joint",
    "RR_calves_Joint",
]


def main():

    if not USD_PATH.exists():
        raise FileNotFoundError(USD_PATH)

    text = USD_PATH.read_text(
        encoding="utf-8"
    )

    joints = re.findall(
        r'def PhysicsRevoluteJoint "([^"]+)"',
        text,
    )

    print("=" * 60)
    print("JETSON DOG USDA INSPECTION")
    print("=" * 60)

    print(f"USD: {USD_PATH}")

    print(f"Revolute Joints: {len(joints)}")

    print()

    for name in joints:
        print(f"  {name}")

    print()

    print("Checking expected joints...")

    for name in EXPECTED_JOINTS:

        if name in joints:
            print(f"[OK] {name}")

        else:
            print(f"[MISSING] {name}")

    print()

    assert len(joints) == 12, (
        f"Expected 12 Revolute Joints, got {len(joints)}"
    )

    assert set(joints) == set(EXPECTED_JOINTS), (
        "Joint names do not match."
    )

    assert "metersPerUnit = 1" in text, (
        "Unexpected USD length unit."
    )

    assert 'upAxis = "Z"' in text, (
        "Unexpected USD up axis."
    )

    print("USDA structural checks passed.")

    print()
    print(
        "Note: This is a text-level check, "
        "not a PhysX dynamics validation."
    )


if __name__ == "__main__":
    main()