# Archaeo-NER-Greek

Archaeology today is confronted with a rapidly increasing volume of data, a large part of which takes the form of textual material. In the case of Greece, which has one of the densest archaeological landscapes worldwide, the volume of excavation reports, catalogues, and scholarly publications is particularly extensive. The systematic management and analysis of this information is becoming increasingly difficult using exclusively traditional methods. A promising approach for extracting structured information from archaeological texts is Named Entity Recognition (NER).

This thesis presents the development of a dataset specifically designed for Greek archaeology. A corpus consisting of 324 sentences was manually annotated by two annotators according to a schema that includes eight entity types: Artefact, Context, Feature, Location, Material, Period, Person, and Species. Inter-Annotator Agreement (IAA) reached an $F_1$-score of 0.91 after the revision of the annotation guidelines, indicating high consistency in the annotation process.

The dataset was used for the evaluation and adaptation of the GLiNER2 model (\texttt{fastino/gliner2-multi-v1}) to the task of recognising archaeological entities in Greek texts, following the same annotation schema. First, the model was evaluated in a zero-shot setting, relying only on the natural-language descriptions of the entity labels. Subsequently, supervised fine-tuning was performed, and the model achieved a final micro-$F_1$ score of 0.65 on an independent, human-annotated test set.

The results showed that fine-tuning improved the model's performance compared with the zero-shot evaluation. At the same time they highlighted the challenges posed by specialised archaeological terminology, multi-word entities, linguistic ambiguity, and the syntactic complexity of Greek archaeological texts. Finally, the use of synthetic data was explored as a pilot method for augmenting the training set in a low-resource setting.

Overall, this thesis aims to create the first specialised resource for the application of NER to Greek archaeology and to provide a basis for future research on the extraction and organisation of archaeological information from Greek texts.

## Dataset

The dataset is available on the Hugging Face Hub: [Stalexan/archaeo-ner-greek](https://huggingface.co/datasets/Stalexan/archaeo-ner-greek/).

## Citation

If you use this dataset or code, please cite the following Master's thesis:

```bibtex
@masterthesis{uoadl:5410546,
    BIBTEX_ENTRY = "masterthesis",
    year = "2026",
    school = "Postgraduate Programme Digital Humanities, National and Kapodistrian University of Athens",
    author = "Alexandropoulou Stavroula",
    title = "Construction and Annotation of a Greek Corpus of Archaeological Texts"
}
```
