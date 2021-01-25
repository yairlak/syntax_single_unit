conda activate mne

1.
sync_log_and_recordings: launch to have the nev files and paradigm output logs synced. May required manual help. 

Check settings in params in
Main/Code/functions/load_settings_params.py
Where you define the patient number and relevant info.


2. Analyze Microphone
- First put Microphone mat file in /Channels/micro and name it CSC0.mat

- Then, run
python generate_multichannel_spectrotemporal_epochs_micro.py --channels 0 --patient 505
which will generate the epochsTFR file in Data/UCLA/patient_???/Epochs/

- Finall, to generate the fig, launch:
python plot_epochs_ERPs.py --patient 505 --channel 0 --tmin -1 --tmax 2 --baseline "(-1, 0)" --sort-key "['chronological_order']" --query "word_position == 1 and block in [2, 4, 6]"

# or, if you want to align to the end and have baseline set:
python plot_epochs_ERPs.py --patient 505 --channel 0 --align end --tmin -2 --tmax 1 --baseline "(0, 1)" --sort-key "['chronological_order']" --block auditory


3. Generate epochTFR (files):
This stage generates an h5 in the /Epochs folder, which will be later used for downward analysis (check the Bash/generate_epochs/ folder for examples of how to parallelize it on your machine or on the cluster using qsub)
Use --help flag to see for more options. 

python generate_multichannel_spectrotemporal_epochs_micro.py -channels 1 -patient 505

4. GAT
This part launches the Generalization Across Time (GAT) decoding pipeline, based on comparisons defined in the comparisons.py file. The comparison.py file contains metadata queries for each comparison (contrast). 
python run_GAT.py -p 479 -p 482 -p 502 -p 504 --cat-k-timepoint 5 -c 0


----------------------------------------------------
------------------------------------------------------
1.  CC  Coordinating conjunction
2.  CD  Cardinal number
3.  DT  Determiner
4.  EX  Existential there
5.  FW  Foreign word
6.  IN  Preposition or subordinating conjunction
7.  JJ  Adjective
8.  JJR Adjective, comparative
9.  JJS Adjective, superlative
10. LS  List item marker
11. MD  Modal
12. NN  Noun, singular or mass
13. NNS Noun, plural
14. NNP Proper noun, singular
15. NNPS    Proper noun, plural
16. PDT Predeterminer
17. POS Possessive ending
18. PRP Personal pronoun
19. PRP$    Possessive pronoun
20. RB  Adverb
21. RBR Adverb, comparative
22. RBS Adverb, superlative
23. RP  Particle
24. SYM Symbol
25. TO  to
26. UH  Interjection
27. VB  Verb, base form
28. VBD Verb, past tense
29. VBG Verb, gerund or present participle
30. VBN Verb, past participle
31. VBP Verb, non-3rd person singular present
32. VBZ Verb, 3rd person singular present
33. WDT Wh-determiner
34. WP  Wh-pronoun
35. WP$ Possessive wh-pronoun
36. WRB Wh-adverb





----------------------------------------------------
-----------------------------------------------------

CC: conjunction, coordinating

& 'n and both but either et for less minus neither nor or plus so
therefore times v. versus vs. whether yet
CD: numeral, cardinal

mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
fifteen 271,124 dozen quintillion DM2,000 ...
DT: determiner

all an another any both del each either every half la many much nary
neither no some such that the them these this those
EX: existential there

there
IN: preposition or conjunction, subordinating

astride among uppon whether out inside pro despite on by throughout
below within for towards near behind atop around if like until below
next into if beside ...
JJ: adjective or numeral, ordinal

third ill-mannered pre-war regrettable oiled calamitous first separable
ectoplasmic battery-powered participatory fourth still-to-be-named
multilingual multi-disciplinary ...
JJR: adjective, comparative

bleaker braver breezier briefer brighter brisker broader bumper busier
calmer cheaper choosier cleaner clearer closer colder commoner costlier
cozier creamier crunchier cuter ...
JJS: adjective, superlative

calmest cheapest choicest classiest cleanest clearest closest commonest
corniest costliest crassest creepiest crudest cutest darkest deadliest
dearest deepest densest dinkiest ...
LS: list item marker

A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
SP-44007 Second Third Three Two * a b c d first five four one six three
two
MD: modal auxiliary

can cannot could couldn't dare may might must need ought shall should
shouldn't will would
NN: noun, common, singular or mass

common-carrier cabbage knuckle-duster Casino afghan shed thermostat
investment slide humour falloff slick wind hyena override subhumanity
machinist ...
NNP: noun, proper, singular

Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
Shannon A.K.C. Meltex Liverpool ...
NNS: noun, common, plural

undergraduates scotches bric-a-brac products bodyguards facets coasts
divestitures storehouses designs clubs fragrances averages
subjectivists apprehensions muses factory-jobs ...
PDT: pre-determiner

all both half many quite such sure this
POS: genitive marker

' 's
PRP: pronoun, personal

hers herself him himself hisself it itself me myself one oneself ours
ourselves ownself self she thee theirs them themselves they thou thy us
PRP$: pronoun, possessive

her his mine my our ours their thy your
RB: adverb

occasionally unabatingly maddeningly adventurously professedly
stirringly prominently technologically magisterially predominately
swiftly fiscally pitilessly ...
RBR: adverb, comparative

further gloomier grander graver greater grimmer harder harsher
healthier heavier higher however larger later leaner lengthier less-
perfectly lesser lonelier longer louder lower more ...
RBS: adverb, superlative

best biggest bluntest earliest farthest first furthest hardest
heartiest highest largest least less most nearest second tightest worst
RP: particle

aboard about across along apart around aside at away back before behind
by crop down ever fast for forth from go high i.e. in into just later
low more off on open out over per pie raising start teeth that through
under unto up up-pp upon whole with you
TO: "to" as preposition or infinitive marker

to
UH: interjection

Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
man baby diddle hush sonuvabitch ...
VB: verb, base form

ask assemble assess assign assume atone attention avoid bake balkanize
bank begin behold believe bend benefit bevel beware bless boil bomb
boost brace break bring broil brush build ...
VBD: verb, past tense

dipped pleaded swiped regummed soaked tidied convened halted registered
cushioned exacted snubbed strode aimed adopted belied figgered
speculated wore appreciated contemplated ...
VBG: verb, present participle or gerund

telegraphing stirring focusing angering judging stalling lactating
hankerin' alleging veering capping approaching traveling besieging
encrypting interrupting erasing wincing ...
VBN: verb, past participle

multihulled dilapidated aerosolized chaired languished panelized used
experimented flourished imitated reunifed factored condensed sheared
unsettled primed dubbed desired ...
VBP: verb, present tense, not 3rd person singular

predominate wrap resort sue twist spill cure lengthen brush terminate
appear tend stray glisten obtain comprise detest tease attract
emphasize mold postpone sever return wag ...
VBZ: verb, present tense, 3rd person singular

bases reconstructs marks mixes displeases seals carps weaves snatches
slumps stretches authorizes smolders pictures emerges stockpiles
seduces fizzes uses bolsters slaps speaks pleads ...
WDT: WH-determiner

that what whatever which whichever
WP: WH-pronoun

that what whatever whatsoever which who whom whosoever
WRB: Wh-adverb

how however whence whenever where whereby whereever wherein whereof why
