# Woopenbaar

Welkom bij de website van Woopenbaar! Deze tool is gemaakt door het AI-studententeam van de TU Delft, [Team Epoch](https://teamepoch.ai).


## Q&A

### Waarom bestaat deze website?
Woopenbaar is gemaakt om makkelijker vrijgegeven overheidsdocumenten te doorzoeken. 

De overheid heeft sinds 2024 een wet genaamd de [WOO](https://www.rijksoverheid.nl/onderwerpen/wet-open-overheid-woo) (Wet Open Overheid).

*"Iedereen heeft recht op informatie over wat de overheid doet, hoe ze dat doet en waarom. Overheidsorganisaties moeten die informatie uit zichzelf geven, of als iemand daarom vraagt. De informatie wordt dan openbaar. Dat is geregeld in de Wet open overheid (Woo). Zo kunnen burgers, maar ook bijvoorbeeld Kamerleden of journalisten, de overheid controleren. Openbaarheid is daarom belangrijk voor de democratie in ons land."* 


Dit betekent elker inwoner van Nederland een WOO-aanvraag kan doen aan een ministerie, en dat deze dan de relevante (gelakte) documenten moet vrijgeven.

Er zijn al veel aanvragen gedaan, en heel veel documenten vrijgegeven. Dit betekent dat er dus al een grote hoeveelheid informatie beschikbaar is!


Helaas is het niet altijd even makkelijk om een antwoord op je vraag te vinden. Hiervoor is Woopenbaar gemaakt.

Met AI doorzoeken wij de vrijgegeven documenten van het ministerie van VWS, en proberen wij zo goed als mogelijk antwoord te geven.


Er zijn een aantal voordelen van deze oplossing:
1) Toegankgelijkheid & transparantie. Documenten zijn makkelijker te doorzoeken en antwoorden dus makkelijker te verkrijgen.
2) Het is een stuk sneller. WOO-aanvragen duren lang om af te handelen, evenals door duizenden (vaak irrelevante) documenten zoeken.
3) Het bespaart mogelijk een hoop geld. Vaak zijn bijna alle documenten die d.m.v. een WOO-aanvraag openbaar gemaakt zouden worden al eerder openbaar gemaakt. Als een groot deel van die vragen al van te voren beantwoord zou worden, zou dat de overheid een hoop geld schelen.


### Welke documenten kan ik doorzoeken?

Momenteel hebben wij 2 databronnen meegenomen.
1) Alle documenten van het [WOO-publicatieplatform](https://open.minvws.nl/) van het ministerie van VWS
2) Meerdere tijdlijnen m.b.t. de Coronacrisis, zoals bijvoorbeeld deze over [COVID-19 adviezen van het OMT en RT](https://www.rivm.nl/corona/adviezen)


### Hoe werkt het?

Deze website is een implementatie van [Retrieval Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation). Kortgezegd houdt dit in dat wij o.b.v. de vraag de meest relevante documenten zoeken, en aan een LLM geven om deze te beantwoorden.

Ook maken wij gebruik van [Adaptive RAG](https://blog.lancedb.com/adaptive-rag/). Dit houdt in dat de LLM zelf bedenkt welke andere vragen relevant zouden zijn om de hoofdvraag te beantwoorden, en hier extra documenten bij zoekt.

### Kan ik er zeker van zijn dat de antwoorden kloppen?

Dat kan helaas niet.

De tool geeft referenties, dus verifieren of het resultaat klopt is zeker mogelijk, maar er is geen manier om zeker te weten dat er niet nog meer relevante documenten bestaan, die niet gevonden zijn.


### Er wordt geen antwoord gegeven op mijn vraag. Hoe kan dit?

Wij hebben een veiligheidslaag ingebouwd. Dit betekent dat irrelevante vragen en vragen die mogelijk schade kunnen opleveren, worden geblokkeerd.

Verder kan het ook dat ons systeem geen relevante documenten kan vinden om de vraag te beantwoorden, of het niet zeker uit de documenten op kan maken. In dit geval zal de LLM vaak weigeren een direct antwoord te geven.


## Privacy disclaimer

Wij zijn de tool nog hard aan het testen. Hiervoor slaan wij **alle vragen** op die aan het systeem gesteld worden. Wij bewaren geen andere informatie van u. 
