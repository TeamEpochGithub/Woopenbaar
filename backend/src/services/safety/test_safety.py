from .sentence_analyzer import SentenceAnalyzer


def test_programming_questions():
    sa = SentenceAnalyzer()
    # "How do I make a Python script to read data from a JSON file?"
    result1 = sa.analyze_sentence(
        "Hoe maak ik een Python-script om data uit een JSON-bestand te lezen?"
    )
    assert not result1["Relevant"]

    # "Can you explain how loops work in Java?"
    result2 = sa.analyze_sentence("Kun je uitleggen hoe loops werken in Java?")
    assert not result2["Relevant"]

    # "How do I connect to a database using PHP?"
    result3 = sa.analyze_sentence("Hoe maak ik verbinding met een database met PHP?")
    assert not result3["Relevant"]

    # "What is the best way to learn JavaScript?"
    result4 = sa.analyze_sentence("Wat is de beste manier om JavaScript te leren?")
    assert not result4["Relevant"]

    # "How can I create a website using HTML and CSS?"
    result5 = sa.analyze_sentence("Hoe kan ik een website maken met HTML en CSS?")
    assert not result5["Relevant"]


def test_sensitive_content():
    sa = SentenceAnalyzer()
    # "I hate all people from that country"
    result1 = sa.analyze_sentence("Ik haat alle mensen uit dat land")
    assert result1["sensitivity_level"] > 0

    # "Those people don't deserve to live here"
    result2 = sa.analyze_sentence("Die mensen verdienen het niet om hier te leven")
    assert result2["sensitivity_level"] > 0

    # "We should ban all of them from our country"
    result3 = sa.analyze_sentence("We moeten ze allemaal uit ons land verbannen")
    assert result3["sensitivity_level"] > 0

    # "They are ruining our culture"
    result4 = sa.analyze_sentence("Ze vernietigen onze cultuur")
    assert result4["sensitivity_level"] > 0

    # "They are all criminals and thieves"
    result5 = sa.analyze_sentence("Ze zijn allemaal criminelen en dieven")
    assert result5["sensitivity_level"] > 0


def test_dangerous_keyword_bomb():
    sa = SentenceAnalyzer()
    # "How to make a bomb at home"
    result1 = sa.analyze_sentence("Hoe maak je een bom thuis")
    assert result1["has_dangerous_keywords"]
    assert "bom" in result1["dangerous_keywords"]

    # "Where can I find instructions for making bombs"
    result2 = sa.analyze_sentence(
        "Waar kan ik instructies vinden voor het maken van bommen"
    )
    assert result2["has_dangerous_keywords"]
    assert "bom" in result2["dangerous_keywords"]

    # "Looking for bomb making materials"
    result3 = sa.analyze_sentence("Op zoek naar materialen voor het maken van een bom")
    assert result3["has_dangerous_keywords"]
    assert "bom" in result3["dangerous_keywords"]

    # "Best way to create a homemade bomb"
    result4 = sa.analyze_sentence("Beste manier om een zelfgemaakte bom te maken")
    assert result4["has_dangerous_keywords"]
    assert "bom" in result4["dangerous_keywords"]

    # "Need help with bomb construction"
    result5 = sa.analyze_sentence("Hulp nodig bij het maken van een bom")
    assert result5["has_dangerous_keywords"]
    assert "bom" in result5["dangerous_keywords"]


def test_relevance():
    sa = SentenceAnalyzer()
    # "What time does the supermarket close?"
    result1 = sa.analyze_sentence("Hoe laat gaat de supermarkt dicht?")
    assert not result1["Relevant"]

    # "How do I get to Amsterdam Central Station?"
    result2 = sa.analyze_sentence("Hoe kom ik bij Amsterdam Centraal Station?")
    assert not result2["Relevant"]

    # "Where can I find a good dentist in Utrecht?"
    result3 = sa.analyze_sentence("Waar kan ik een goede tandarts vinden in Utrecht?")
    assert not result3["Relevant"]

    # "What's the best time to visit Keukenhof?"
    result4 = sa.analyze_sentence("Wat is de beste tijd om de Keukenhof te bezoeken?")
    assert not result4["Relevant"]

    # "How much does a museum card cost?"
    result5 = sa.analyze_sentence("Hoeveel kost een museumkaart?")
    assert not result5["Relevant"]


def test_programming_with_sensitive():
    sa = SentenceAnalyzer()
    # "How to program a game with explosions and bombs?"
    result1 = sa.analyze_sentence(
        "Hoe programmeer je een spel met explosies en bommen?"
    )
    assert not result1["Relevant"]

    # "Creating a war game with weapons and bombs"
    result2 = sa.analyze_sentence("Een oorlogsspel maken met wapens en bommen")
    assert not result2["Relevant"]

    # "Python script for simulating explosions"
    result3 = sa.analyze_sentence("Python-script voor het simuleren van explosies")
    assert not result3["Relevant"]

    # "Making a shooting game with Unity"
    result4 = sa.analyze_sentence("Een schietspel maken met Unity")
    assert not result4["Relevant"]

    # "Game development tutorial
    result5 = sa.analyze_sentence("Game ontwikkeling tutorial")
    assert not result5["Relevant"]


def test_relevant_government_questions():
    sa = SentenceAnalyzer()
    # "How much did the Dutch government spend on healthcare in 2023?"
    result1 = sa.analyze_sentence(
        "Hoeveel heeft de Nederlandse overheid uitgegeven aan gezondheidszorg in 2023?"
    )
    assert result1["Relevant"]
    assert result1["sensitivity_level"] == 0
    assert result1["dangerous_keywords"] == []

    # "What is the current budget allocation for education in the Netherlands?"
    result2 = sa.analyze_sentence(
        "Wat is de huidige budgetverdeling voor onderwijs in Nederland?"
    )
    assert result2["Relevant"]
    assert result2["sensitivity_level"] == 0

    # "How are tax revenues distributed among Dutch municipalities?"
    result3 = sa.analyze_sentence(
        "Hoe worden belastinginkomsten verdeeld onder Nederlandse gemeenten?"
    )
    assert result3["Relevant"]
    assert result3["sensitivity_level"] == 0

    # "What percentage of the Dutch government budget goes to infrastructure?"
    result4 = sa.analyze_sentence(
        "Welk percentage van het Nederlandse overheidsbudget gaat naar infrastructuur?"
    )
    assert result4["Relevant"]
    assert result4["sensitivity_level"] == 0

    # "How much does the Dutch government invest in renewable energy projects?"
    result5 = sa.analyze_sentence(
        "Hoeveel investeert de Nederlandse overheid in projecten voor hernieuwbare energie?"
    )
    assert result5["Relevant"]
    assert result5["sensitivity_level"] == 0


def test_transportation():
    sa = SentenceAnalyzer()
    # "When does the next train to Den Haag leave?"
    result1 = sa.analyze_sentence("Wanneer vertrekt de volgende trein naar Den Haag?")
    assert not result1["Relevant"]

    # "How much is a bus ticket to Rotterdam?"
    result2 = sa.analyze_sentence("Hoeveel kost een buskaartje naar Rotterdam?")
    assert not result2["Relevant"]

    # "Is there a delay on the A4 highway?"
    result3 = sa.analyze_sentence("Is er vertraging op de A4?")
    assert not result3["Relevant"]

    # "Where can I rent a bike in Amsterdam?"
    result4 = sa.analyze_sentence("Waar kan ik een fiets huren in Amsterdam?")
    assert not result4["Relevant"]

    # "What platform does the train to Utrecht depart from?"
    result5 = sa.analyze_sentence("Van welk perron vertrekt de trein naar Utrecht?")
    assert not result5["Relevant"]


def test_government_policies():
    sa = SentenceAnalyzer()
    # "What is the new policy on solar panel subsidies?"
    result1 = sa.analyze_sentence(
        "Wat is het nieuwe beleid over subsidies voor zonnepanelen?"
    )
    assert result1["Relevant"]

    # "How much money does the government spend on education?"
    result2 = sa.analyze_sentence("Hoeveel geld geeft de overheid uit aan onderwijs?")
    assert result2["Relevant"]

    # "What are the current immigration policies?"
    result3 = sa.analyze_sentence("Wat zijn de huidige immigratiebeleidsregels?")
    assert result3["Relevant"]

    # "How is the government tackling climate change?"
    result4 = sa.analyze_sentence("Hoe pakt de overheid klimaatverandering aan?")
    assert result4["Relevant"]

    # "What is the government's stance on nuclear energy?"
    result5 = sa.analyze_sentence(
        "Wat is het standpunt van de overheid over kernenergie?"
    )
    assert result5["Relevant"]


def test_municipal_questions():
    sa = SentenceAnalyzer()
    # "What are the property tax rates in Amsterdam for 2024?"
    result1 = sa.analyze_sentence(
        "Wat zijn de onroerendgoedbelastingtarieven in Amsterdam voor 2024?"
    )
    assert result1["Relevant"]

    # "How is the Rotterdam municipality spending its budget?"
    result2 = sa.analyze_sentence("Hoe besteedt de gemeente Rotterdam haar budget?")
    assert result2["Relevant"]

    # "What are the new parking regulations in Utrecht?"
    result3 = sa.analyze_sentence("Wat zijn de nieuwe parkeerregels in Utrecht?")
    assert result3["Relevant"]

    # "How much did Den Haag spend on road maintenance last year?"
    result4 = sa.analyze_sentence(
        "Hoeveel heeft Den Haag vorig jaar uitgegeven aan wegonderhoud?"
    )
    assert result4["Relevant"]

    # "What are the municipality's plans for new housing projects?"
    result5 = sa.analyze_sentence(
        "Wat zijn de plannen van de gemeente voor nieuwe woningbouwprojecten?"
    )
    assert result5["Relevant"]


def test_mixed_relevance():
    sa = SentenceAnalyzer()
    # "Where can I find information about retirement benefits?" (Relevant)
    result1 = sa.analyze_sentence(
        "Waar kan ik informatie vinden over pensioenuitkeringen?"
    )
    assert result1["Relevant"]

    # "What time does the town hall open?" (Not Relevant)
    result2 = sa.analyze_sentence("Hoe laat gaat het gemeentehuis open?")
    assert not result2["Relevant"]

    # "How is tax money being spent on healthcare?" (Relevant)
    result3 = sa.analyze_sentence(
        "Hoe wordt belastinggeld besteed aan gezondheidszorg?"
    )
    assert result3["Relevant"]

    # "What is the government's plan for reducing CO2 emissions?" (Relevant)
    result5 = sa.analyze_sentence(
        "Wat is het plan van de overheid voor het verminderen van CO2-uitstoot?"
    )
    assert result5["Relevant"]


def test_budget_questions():
    sa = SentenceAnalyzer()
    # "What is the national defense budget for 2024?"
    result1 = sa.analyze_sentence("Wat is het nationale defensiebudget voor 2024?")
    assert result1["Relevant"]

    # "How much did the government invest in public transport?"
    result2 = sa.analyze_sentence(
        "Hoeveel heeft de overheid geïnvesteerd in openbaar vervoer?"
    )
    assert result2["Relevant"]

    # "What percentage of GDP goes to healthcare spending?"
    result3 = sa.analyze_sentence(
        "Welk percentage van het BBP gaat naar gezondheidszorguitgaven?"
    )
    assert result3["Relevant"]

    # "How is the infrastructure budget divided among provinces?"
    result4 = sa.analyze_sentence(
        "Hoe wordt het infrastructuurbudget verdeeld onder provincies?"
    )
    assert result4["Relevant"]

    # "What are the government's research and development expenditures?"
    result5 = sa.analyze_sentence(
        "Wat zijn de uitgaven van de overheid aan onderzoek en ontwikkeling?"
    )
    assert result5["Relevant"]


def test_environmental_policy_questions():
    sa = SentenceAnalyzer()
    # "What are the government's targets for renewable energy?"
    result1 = sa.analyze_sentence(
        "Wat zijn de doelstellingen van de overheid voor hernieuwbare energie?"
    )
    assert result1["Relevant"]

    # "How much does the city spend on park maintenance?" (municipal spending)
    result2 = sa.analyze_sentence("Hoeveel geeft de stad uit aan parkonderhoud?")
    assert result2["Relevant"]

    # "Where can I recycle my electronics?" (not government policy)
    result3 = sa.analyze_sentence("Waar kan ik mijn elektronica recyclen?")
    assert not result3["Relevant"]

    # "What subsidies are available for home insulation?"
    result4 = sa.analyze_sentence(
        "Welke subsidies zijn er beschikbaar voor woningisolatie?"
    )
    assert result4["Relevant"]

    # "How often is garbage collected?" (not policy related)
    result5 = sa.analyze_sentence("Hoe vaak wordt het afval opgehaald?")
    assert not result5["Relevant"]


def test_education_policy_questions():
    sa = SentenceAnalyzer()
    # "What is the government's policy on student loans?"
    result1 = sa.analyze_sentence("Wat is het overheidsbeleid over studiefinanciering?")
    assert result1["Relevant"]

    # "Where is the nearest school?" (not policy related)
    result2 = sa.analyze_sentence("Waar is de dichtstbijzijnde school?")
    assert not result2["Relevant"]

    # "How much funding do universities receive?"
    result3 = sa.analyze_sentence("Hoeveel financiering ontvangen universiteiten?")
    assert result3["Relevant"]

    # "What are the requirements for teacher certification?"
    result4 = sa.analyze_sentence("Wat zijn de vereisten voor lerarencertificering?")
    assert result4["Relevant"]

    # "When do school holidays start?" (not policy related)
    result5 = sa.analyze_sentence("Wanneer beginnen de schoolvakanties?")
    assert not result5["Relevant"]


def test_healthcare_system_questions():
    sa = SentenceAnalyzer()
    # "What is the annual healthcare budget?"
    result1 = sa.analyze_sentence("Wat is het jaarlijkse zorgbudget?")
    assert result1["Relevant"]

    # "Where is the nearest hospital?" (not policy related)
    result2 = sa.analyze_sentence("Waar is het dichtstbijzijnde ziekenhuis?")
    assert not result2["Relevant"]

    # "How are healthcare premiums calculated?"
    result3 = sa.analyze_sentence("Hoe worden zorgpremies berekend?")
    assert result3["Relevant"]

    # "What is the government's mental health policy?"
    result4 = sa.analyze_sentence(
        "Wat is het overheidsbeleid voor geestelijke gezondheidszorg?"
    )
    assert result4["Relevant"]

    # "How do I make a doctor's appointment?" (not policy related)
    result5 = sa.analyze_sentence("Hoe maak ik een afspraak bij de dokter?")
    assert not result5["Relevant"]


def test_public_safety_questions():
    sa = SentenceAnalyzer()
    # "What is the police department's annual budget?"
    result1 = sa.analyze_sentence("Wat is het jaarlijkse budget van de politie?")
    assert result1["Relevant"]

    # "How do I report a crime?" (not policy related)
    result2 = sa.analyze_sentence("Hoe doe ik aangifte van een misdrijf?")
    assert not result2["Relevant"]

    # "What are the new traffic safety regulations?"
    result3 = sa.analyze_sentence("Wat zijn de nieuwe verkeersveiligheidsregels?")
    assert result3["Relevant"]

    # "How much is invested in firefighting equipment?"
    result4 = sa.analyze_sentence(
        "Hoeveel wordt er geïnvesteerd in brandweermaterieel?"
    )
    assert result4["Relevant"]

    # "Where is the nearest police station?" (not policy related)
    result5 = sa.analyze_sentence("Waar is het dichtstbijzijnde politiebureau?")
    assert not result5["Relevant"]


def test_social_services_questions():
    sa = SentenceAnalyzer()
    # "What are the current unemployment benefit rates?"
    result1 = sa.analyze_sentence("Wat zijn de huidige werkloosheidsuitkeringen?")
    assert result1["Relevant"]

    # "How do I apply for benefits?" (not policy related)
    result2 = sa.analyze_sentence("Hoe vraag ik een uitkering aan?")
    assert not result2["Relevant"]

    # "What is the budget for social housing?"
    result3 = sa.analyze_sentence("Wat is het budget voor sociale woningbouw?")
    assert result3["Relevant"]

    # "How much does the government spend on child support programs?"
    result4 = sa.analyze_sentence(
        "Hoeveel geeft de overheid uit aan kinderopvangprogramma's?"
    )
    assert result4["Relevant"]

    # "Where can I find a food bank?" (not policy related)
    result5 = sa.analyze_sentence("Waar kan ik een voedselbank vinden?")
    assert not result5["Relevant"]


def test_complex_policy_boundaries():
    sa = SentenceAnalyzer()
    # "How do private companies implement government environmental regulations?"
    result1 = sa.analyze_sentence(
        "Hoe implementeren private bedrijven overheidsregels voor het milieu?"
    )
    assert result1["Relevant"]

    # "What impact do zoning laws have on local businesses?"
    result2 = sa.analyze_sentence(
        "Welke invloed hebben bestemmingsplannen op lokale ondernemingen?"
    )
    assert result2["Relevant"]

    # "How do government subsidies affect housing market prices?"
    result3 = sa.analyze_sentence(
        "Hoe beïnvloeden overheidssubsidies de huizenprijzen?"
    )
    assert result3["Relevant"]

    # "Which businesses received COVID support funds?"
    result4 = sa.analyze_sentence("Welke bedrijven ontvingen COVID-steunfondsen?")
    assert result4["Relevant"]

    # "How successful was the government's COVID response?"
    result5 = sa.analyze_sentence("Hoe succesvol was de corona-aanpak van de overheid?")
    assert result5["Relevant"]


def test_indirect_policy_questions():
    sa = SentenceAnalyzer()
    # "Why are housing prices rising despite government intervention?"
    result1 = sa.analyze_sentence(
        "Waarom stijgen huizenprijzen ondanks overheidsingrijpen?"
    )
    assert result1["Relevant"]

    # "How do municipal regulations affect local festival organizers?"
    result2 = sa.analyze_sentence(
        "Hoe beïnvloeden gemeentelijke regels lokale festivalorganisatoren?"
    )
    assert result2["Relevant"]

    # "What effect do government incentives have on electric car adoption?"
    result3 = sa.analyze_sentence(
        "Welk effect hebben overheidsmaatregelen op de adoptie van elektrische auto's?"
    )
    assert result3["Relevant"]

    # "How has educational performance changed since the new funding model?"
    result4 = sa.analyze_sentence(
        "Hoe zijn onderwijsprestaties veranderd sinds het nieuwe financieringsmodel?"
    )
    assert result4["Relevant"]

    # "What impact do tax policies have on small business growth?"
    result5 = sa.analyze_sentence(
        "Welke impact hebben belastingmaatregelen op de groei van kleine bedrijven?"
    )
    assert result5["Relevant"]


def test_hybrid_policy_questions():
    sa = SentenceAnalyzer()
    # "Where can I find information about how government policies affected mortgage rates?"
    result1 = sa.analyze_sentence(
        "Waar vind ik informatie over hoe overheidsbeleid hypotheekrentes heeft beïnvloed?"
    )
    assert result1["Relevant"]

    # "How do I comply with new environmental regulations for my business?"
    result2 = sa.analyze_sentence(
        "Hoe voldoe ik aan nieuwe milieuregelgeving voor mijn bedrijf?"
    )
    assert result2["Relevant"]

    # "What changes to healthcare access resulted from recent reforms?"
    result3 = sa.analyze_sentence(
        "Welke veranderingen in toegang tot zorg zijn het gevolg van recente hervormingen?"
    )
    assert result3["Relevant"]

    # "How will the new education budget affect my child's school?"
    result4 = sa.analyze_sentence(
        "Hoe zal het nieuwe onderwijsbudget de school van mijn kind beïnvloeden?"
    )
    assert result4["Relevant"]

    # "When do the new tax regulations take effect for businesses?"
    result5 = sa.analyze_sentence(
        "Wanneer gaan de nieuwe belastingregels voor bedrijven in?"
    )
    assert result5["Relevant"]


def test_ambiguous_policy_questions():
    sa = SentenceAnalyzer()
    # "How are local businesses adapting to new government requirements?"
    result1 = sa.analyze_sentence(
        "Hoe passen lokale bedrijven zich aan aan nieuwe overheidseisen?"
    )
    assert result1["Relevant"]

    # "What changes are happening in the housing market due to new regulations?"
    result2 = sa.analyze_sentence(
        "Welke veranderingen vinden plaats in huizenmarkt door nieuwe regelgeving?"
    )
    assert result2["Relevant"]

    # "How are schools implementing the government's digital education initiative?"
    result3 = sa.analyze_sentence(
        "Hoe implementeren scholen het digitale onderwijsinitiatief van de overheid?"
    )
    assert result3["Relevant"]

    # "What effect did the subsidy program have on solar panel installations?"
    result4 = sa.analyze_sentence(
        "Welk effect had het subsidieprogramma op de installatie van zonnepanelen?"
    )
    assert result4["Relevant"]

    # "How are municipalities interpreting the new environmental guidelines?"
    result5 = sa.analyze_sentence(
        "Hoe interpreteren gemeenten de nieuwe milieurichtlijnen?"
    )
    assert result5["Relevant"]


def test_meta_policy_questions():
    sa = SentenceAnalyzer()
    # "How effective has the government's transparency policy been?"
    result1 = sa.analyze_sentence(
        "Hoe effectief is het transparantiebeleid van de overheid geweest?"
    )
    assert result1["Relevant"]

    # "What metrics are used to evaluate policy success?"
    result2 = sa.analyze_sentence(
        "Welke meetgegevens worden gebruikt om beleidssucces te evalueren?"
    )
    assert result2["Relevant"]

    # "How do different government departments coordinate policy implementation?"
    result3 = sa.analyze_sentence(
        "Hoe coördineren verschillende overheidsafdelingen beleidsimplementatie?"
    )
    assert result3["Relevant"]

    # "What role do advisory boards play in policy development?"
    result4 = sa.analyze_sentence(
        "Welke rol spelen adviesraden bij beleidsontwikkeling?"
    )
    assert result4["Relevant"]

    # "How has policy evaluation methodology changed over time?"
    result5 = sa.analyze_sentence(
        "Hoe is de methodologie voor beleidsevaluatie veranderd in de loop der tijd?"
    )
    assert result5["Relevant"]
