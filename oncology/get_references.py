from grobid_client.grobid_client import GrobidClient
from lxml import etree
from glob import glob

IN_DIR = "pdfs"
OUT_DIR = "test_out"

if __name__ == "__main__":
    client = GrobidClient(config_path="config.json")
    client.process("processReferences", IN_DIR, output=OUT_DIR)
    xml_files = glob(f'{OUT_DIR}/*.xml')
    if len(xml_files) > 0:
        tree = etree.parse(xml_files[0]) #process the first file
        root = tree.getroot()
        ns = {"tei": root.nsmap.get(None)}

        results = []

        for bibl in root.xpath('//tei:biblStruct', namespaces=ns):
            bibl_id = bibl.get('{http://www.w3.org/XML/1998/namespace}id') or bibl.get('id')

            analytic_titles = bibl.xpath('./tei:analytic/tei:title', namespaces=ns)
            if analytic_titles:
                title = analytic_titles[0].text
            else:
                monogr_titles = bibl.xpath('./tei:monogr/tei:title', namespaces=ns)
                title = monogr_titles[0].text

            results.append((bibl_id, title))


        for bibl_id, title in results:
            print(f"ID: {bibl_id} -> Title: {title}")
