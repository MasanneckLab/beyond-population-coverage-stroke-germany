from pathlib import Path
from hashlib import sha256
import html
import re
import unicodedata
import xml.etree.ElementTree as ET

def norm(value):
    text = html.unescape(str(value)).casefold().replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.replace("strasse", "str")
    return re.sub(r"[^a-z0-9]", "", text)


def contact(path):
    root = ET.parse(path).getroot()
    site_id = path.name.split("-")[1]
    sites = root.findall(".//Standortkontaktdaten")
    exact = [s for s in sites if (s.findtext("Standortnummer") or "").strip() == site_id]
    if len(exact) == 1:
        element, source = exact[0], "site_contact_matching_filename"
    elif len(sites) == 1:
        element, source = sites[0], "single_site_contact"
    elif not sites:
        element, source = root.find(".//Krankenhauskontaktdaten"), "single_hospital_contact"
    else:
        raise ValueError(f"Ambiguous site contact: {path.name}")
    if element is None:
        raise ValueError(f"No contact in {path.name}")
    access = element.find("Kontakt_Zugang")
    if access is None:
        access = element
    values = {k: html.unescape((access.findtext(".//"+tag) or "").strip()) for k,tag in
              [("street","Strasse"),("house_number","Hausnummer"),("postal_code","Postleitzahl"),("city","Ort")]}
    values["site_name"] = html.unescape((element.findtext("Name") or "").strip())
    values["source_contact"] = source
    values["official_url"] = access.findtext("URL_Zugang") or root.findtext(".//URL_Internetseite_Krankenhaus") or ""
    values["xml_sha256"] = sha256(path.read_bytes()).hexdigest()
    values["site_address"] = f"{values['street']} {values['house_number']}, {values['postal_code']} {values['city']}"
    return values
