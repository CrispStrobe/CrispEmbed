#!/usr/bin/env python3
"""Render authored ground truth; frozen PNGs, not worker rerenders, are the inputs."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __version__

HERE = Path(__file__).resolve().parent
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BOLD = FONT.with_name('DejaVuSans-Bold.ttf')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cases = [
        dict(id='field_report', title='FIELD REPORT 47', lines=[
            'Station: North Quay    Date: 13 November 2026',
            '',
            'At 07:35 the survey team opened the east gate.',
            'Three blue crates stood beside the loading ramp.',
            'Crate A17 held 24 brass fittings; crate B09 held 18.',
            'The final crate was empty and its seal was intact.',
            '',
            'Mira checked the gauge twice: 12.8 kPa at 08:10',
            'and 12.6 kPa at 08:25. No leak was observed.',
            'The team replaced one worn washer, tightened',
            'four bolts, and marked the panel READY.',
            '',
            'Next inspection: 20 November 2026 at 09:00.',
            'Keep the west door closed during the pressure test.',
        ], tables=[]),
        dict(id='invoice_grid', title='HARBOUR SUPPLY INVOICE', lines=[
            'Invoice: HS-0719    Date: 13 November 2026',
            'Customer: North Quay Workshop',
        ], tables=[[['Item', 'Qty', 'Unit EUR', 'Total EUR'],
                    ['Brass fitting', '24', '3.50', '84.00'],
                    ['Steel washer', '18', '0.25', '4.50'],
                    ['Safety label', '6', '1.20', '7.20']]],
             footer=['Subtotal: EUR 95.70', 'Delivery: EUR 4.30',
                     'Amount due: EUR 100.00', 'Payment reference: HS-0719']),
        dict(id='accented_dispatch', title='EXPÉDITION 28', lines=[
            'Destinataire: Chloé Martin',
            'Adresse: 18 rue des Érables, 69002 Lyon',
            'Référence: FR-028-731',
            '',
            'Contenu: café moulu, thé vert et cacao.',
            'Quantité: 12 boîtes; poids net: 8,75 kg.',
            'Départ prévu: 16 novembre 2026 à 14 h 30.',
            '',
            'Empfänger: Jürgen Weiß',
            'Straße: Große Allee 7, 10115 Berlin',
            'Hinweis: Kühl und trocken lagern.',
            'Prüfung: Größe M; Stückzahl 12; Maß 30 cm.',
            '',
            'Entrega: São Luís, código BR-024.',
            'Observação: açúcar, limão e maçã.',
        ], tables=[]),
    ]
    manifest = dict(schema='lighton4b-authored-quality-fixtures/v1', license='MIT',
        annotation='New text authored for this evaluation before any model outputs; synthetic clean pages.',
        renderer=dict(pillow=__version__, font_sha256=sha(FONT), bold_font_sha256=sha(BOLD)), cases=[])
    for case in cases:
        image = Image.new('RGB', (1000, 1000), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(FONT), 25)
        titlefont = ImageFont.truetype(str(BOLD), 32)
        draw.text((60, 50), case['title'], fill='black', font=titlefont)
        y = 125
        for line in case['lines']:
            if line:
                assert draw.textbbox((60,y),line,font=font)[2] < 940, line
                draw.text((60,y), line, fill='black', font=font)
            y += 40
        text = case['title'] + '\n' + '\n'.join(case['lines'])
        if case['tables']:
            y += 25
            xs = [60, 420, 540, 730, 940]
            grid = case['tables'][0]
            for ri, row in enumerate(grid):
                top = y + ri * 64
                for ci, cell in enumerate(row):
                    draw.rectangle((xs[ci], top, xs[ci+1], top+64), outline='black', width=2)
                    draw.text((xs[ci]+12, top+17), cell, fill='black', font=font)
            y += len(grid)*64 + 45
            text += '\n' + '\n'.join(' '.join(row) for row in grid)
            for line in case['footer']:
                draw.text((60,y), line, fill='black', font=font)
                y += 40
            text += '\n' + '\n'.join(case['footer'])
        assert y < 960
        png = HERE/(case['id']+'.png')
        image.save(png, optimize=False)
        manifest['cases'].append(dict(id=case['id'], image=png.name, width=image.width, height=image.height,
            png_sha256=sha(png), rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            gold=dict(text=text, tables=case['tables'])))
    (HERE/'fixtures.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')

if __name__ == '__main__':
    main()
