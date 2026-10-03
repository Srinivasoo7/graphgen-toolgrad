from crux.branding import ONTOLOGY_MODULE, PRODUCT, TOKEN_MODULE
from crux.openapi import openapi_document


def test_module_names_are_user_facing():
    assert PRODUCT == "Crux"
    assert "Enterprise Ontology" in ONTOLOGY_MODULE
    assert "Utopia" in ONTOLOGY_MODULE
    assert "Token Optimization" in TOKEN_MODULE
    assert "Headroom" in TOKEN_MODULE


def test_openapi_describes_both_modules():
    spec = openapi_document()
    blob = spec["info"]["title"] + " " + spec["info"]["description"]
    assert PRODUCT in blob
    assert "Enterprise Ontology" in blob
    assert "Token Optimization" in blob
