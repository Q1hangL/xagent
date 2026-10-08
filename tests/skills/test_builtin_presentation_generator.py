from pathlib import Path

from xagent.skills.parser import SkillParser


def _presentation_generator_content() -> str:
    skill_dir = (
        Path(__file__).parents[2]
        / "src"
        / "xagent"
        / "skills"
        / "builtin"
        / "presentation-generator"
    )
    return " ".join(SkillParser.parse(skill_dir)["content"].split())


def test_google_slides_guidance_passes_default_slide_id_only_when_returned() -> None:
    content = _presentation_generator_content()

    assert "always pass the `default_slide_id`" not in content
    assert (
        "when `google_slides_create_presentation` returns a `default_slide_id`"
        in content
    )
    assert "it is `null` when there is no default page to replace" in content
    assert "the returned `default_slide_id`, when one is returned," in content
