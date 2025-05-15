"""
Custom configuration for the Marker document processing system.

This module customizes the behavior of Marker components by:
1. Overriding the layout processing method
2. Providing Dutch-language prompts for:
   - Layout block relabeling
   - Complex region processing
   - Image description generation

Assumptions:
    - Ollama service is running locally on port 11434
    - Dutch language processing is required
    - Images may contain redacted text marked with '5.1.2e'
    - Layout processing should be done in batches
"""

from typing import List

from marker.builders.llm_layout import LLMLayoutBuilder
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.schema.document import PageGroup
from surya.layout import LayoutResult


def custom_surya_layout(self, pages: List[PageGroup]) -> List[LayoutResult]:
    """
    Override the default Surya layout processing method.

    Customizations:
        - Uses configurable batch size (defaults to 1)
        - Supports progress bar toggling
        - Processes high-resolution images

    Args:
        pages: List of page groups to process

    Returns:
        List of layout analysis results
    """
    batch_size = getattr(self, "batch_size", 1)
    self.layout_model.disable_tqdm = self.disable_tqdm
    layout_results = self.layout_model(
        [p.get_image(highres=True) for p in pages], batch_size=int(batch_size)
    )
    return layout_results


custom_topk_relabelling_prompt = """U bent een lay-outexpert gespecialiseerd in documentanalyse.
Uw taak is om lay-outblokken in afbeeldingen opnieuw te labelen om de nauwkeurigheid van een bestaand lay-outmodel te verbeteren.
U krijgt een afbeelding van een lay-outblok en de top k voorspellingen van het huidige model, samen met de betrouwbaarheidsscores per label.
Uw taak is om de afbeelding te analyseren en het meest geschikte label te kiezen uit de aangeboden top k voorspellingen.
Verzin geen nieuwe labels.
Bekijk de afbeelding zorgvuldig en overweeg de gegeven voorspellingen. Houd rekening met de betrouwbaarheidsscores van het model. De betrouwbaarheid wordt gerapporteerd op een schaal van 0-1, waarbij 1 100% betrouwbaar is. Als het bestaande label het meest geschikt is, moet u het niet wijzigen.

**Instructies**
1. Analyseer de afbeelding en overweeg de gegeven top k voorspellingen.
2. Schrijf een korte beschrijving van de afbeelding en welk van de potentiële labels volgens u de meest nauwkeurige weergave is van het lay-outblok.
3. Kies het meest geschikte label uit de gegeven top k voorspellingen.

Hier zijn beschrijvingen van de lay-outblokken waaruit u kunt kiezen:

{potential_labels}

Hier zijn de top k voorspellingen van het model:

{top_k}
"""
custom_complex_relabeling_prompt = """U bent een lay-outexpert gespecialiseerd in documentanalyse.
Uw taak is om lay-outblokken in afbeeldingen opnieuw te labelen om de nauwkeurigheid van een bestaand lay-outmodel te verbeteren.
U krijgt een afbeelding van een lay-outblok en enkele potentiële labels die geschikt zouden kunnen zijn.
Uw taak is om de afbeelding te analyseren en het meest geschikte label te kiezen uit de aangeboden labels.
Verzin geen nieuwe labels.

**Instructies**
1. Analyseer de afbeelding en overweeg de potentiële labels.
2. Schrijf een korte beschrijving van de afbeelding en welk van de potentiële labels volgens u de meest nauwkeurige weergave is van het lay-outblok.
3. Kies het meest geschikte label uit de aangeboden labels.

Potentiële labels:

{potential_labels}

Reageer alleen met een van `Figure`, `Picture`, `ComplexRegion`, `Table`, of `Form`.
"""

custom_image_description_prompt = """ U bent een documentanalyse-expert voor de Nederlandse overheid, gespecialiseerd in het maken van tekstbeschrijvingen voor afbeeldingen uit interne communicatiedocumenten.
U ontvangt een afbeelding van een foto, figuur, screenshot of email. Uw taak is om een korte beschrijving van de afbeelding in het Nederlands te maken. Er is een mogelijkheid dat de tekst in de afbeelding grijs gelakte tekst bevat met 5.1.2e als referentie, dit waren persoonsgegevens en kunnen worden weergegeven als `5.1.2e`.

**Instructies:**
1. Bekijk de afbeelding zorgvuldig.
2. Analyseer eventuele tekst die uit de afbeelding is geëxtraheerd.
3. Maak een duidelijke en professionele Nederlandse beschrijving die:
   - Het type inhoud beschrijft (tabel, grafiek, diagram, screenshot, etc.)
   - De belangrijkste zichtbare elementen uitlegt
   - Eventuele relevante getallen of datapunten bevat
   - Context geeft over het doel van het document
4. De beschrijving moet in het Nederlands zijn en 3-4 zinnen lang zijn.

**Voorbeeld:**
Input:
``text
"Fruitvoorkeur Onderzoek"
20, 15, 10
Appels, Bananen, Sinaasappels

Output:
Deze grafiek toont de resultaten van een fruitvoorkeursonderzoek met drie verschillende soorten fruit. De staafdiagram laat zien dat appels het meest populair zijn met 20 stemmen, gevolgd door bananen met 15 stemmen en sinaasappels met 10 stemmen. De gegevens zijn duidelijk weergegeven met het aantal voorkeuren op de y-as en de fruitsoorten op de x-as. De visualisatie maakt effectief gebruik van verschillende hoogtes om de verschillen in voorkeuren aan te tonen.

**Input:**
``text
{raw_text}
``
"""

custom_complex_region_prompt = """U bent een tekstformatteringsspecialist met expertise in het nauwkeurig reproduceren van complexe documentlayouts.

U ontvangt een afbeelding van een tekstblok samen met de ruwe tekst die uit die afbeelding is geëxtraheerd. Uw taak is om correct geformatteerde markdown te genereren die de inhoud en structuur van de originele afbeelding getrouw weergeeft.

**Formatteringsrichtlijnen:**
- Gebruik * voor cursief, ** voor vet, en ` voor inline code
- Formatteer koppen met # (één # voor hoofdkoppen, tot zes # voor subkoppen)
- Maak lijsten met - voor opsommingstekens of 1. voor genummerde lijsten
- Formatteer links als [tekst](url)
- Gebruik ``` voor codeblokken
- Formatteer wiskundige expressies met <math>inline formule</math> of <math display="block">display formule</math>
- Maak tabellen met markdown tabelsyntax met | en - karakters
- Voor formuliervelden, maak tabellen met "Label" en "Waarde" kolommen
- Verwijder overtollige witruimte tussen tekst

**Proces:**
1. Bekijk de aangeleverde afbeelding zorgvuldig
2. Controleer de geëxtraheerde tekst op nauwkeurigheid en formatteringsproblemen
3. Maak een markdown-weergave die alle inhoud en opmaak behoudt
4. Zorg ervoor dat tabellen correct zijn uitgelijnd en gestructureerd
5. Ruim formatteringsonregelmatigheden op, maar behoud de oorspronkelijke betekenis

**Voorbeeld 1:**
Input:
```text
Tabel 1: Autoverkopen
```
Output:
```markdown
## Tabel 1: Autoverkopen

| **Auto** | **Verkopen** |
|----------|--------------|
| Honda    | 100          |
| Toyota   | 200          |
```

**Voorbeeld 2:**
Input met overtollige witruimte in referentienummer:
```text
Neem contact op met Jan via     5.1.2e   

voor meer informatie.
```
Output met opgeschoond referentienummer:
```markdown
Neem contact op met Jan via `5.1.2e` voor meer informatie.
```

**Input:**
```text
{extracted_text}
```
"""
LLMLayoutBuilder.surya_layout = custom_surya_layout
LLMLayoutBuilder.topk_relabelling_prompt = custom_topk_relabelling_prompt
LLMLayoutBuilder.complex_relabeling_prompt = custom_complex_relabeling_prompt
LLMImageDescriptionProcessor.image_description_prompt = custom_image_description_prompt
LLMComplexRegionProcessor.complex_region_prompt = custom_complex_region_prompt
