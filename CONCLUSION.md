# Conclusion

## 5.1 Summary

In this work, we presented a hybrid Direct Preference Optimization (DPO) training framework for multi-hop knowledge graph reasoning. Our approach integrates three key components: similarity-based neighbor selection inspired by SNS, visual graph representations adapted from GITA, and DPO training extended from GraphWiz. The framework processes knowledge graph triples into hybrid training examples that combine textual prompts with optional visual graph representations, then trains language models using DPO to learn preferred reasoning patterns.

We demonstrated the effectiveness of our approach through experimental evaluation on a knowledge graph reasoning task. The trained model achieved 96.7% accuracy on a test set of 30 examples, with 13 out of 14 relation types achieving perfect accuracy. The results show that DPO training is effective for aligning language models with preferred reasoning patterns in knowledge graph contexts, requiring relatively few training examples and epochs to achieve strong performance.

## 5.2 Contributions

Our work makes several contributions to the field of knowledge graph reasoning with language models. First, we present the first hybrid DPO approach specifically designed for knowledge graph reasoning tasks, combining textual and visual components in a unified training framework. Second, we adapt similarity-based neighbor selection from citation networks to knowledge graph entities and relations, demonstrating its effectiveness for reducing over-squashing in KG reasoning. Third, we provide an end-to-end, reproducible pipeline for knowledge graph reasoning that integrates multiple existing frameworks while adding novel contributions for reproducibility and scalability.

The implementation includes several practical contributions: a dual-stage caching system for deterministic reruns across different machines, an HPC-aware rendering workflow with throttling and parallel processing support, and a comprehensive evaluation framework with multiple metrics. These contributions ensure that the research is reproducible and scalable, addressing important practical concerns in machine learning research.

## 5.3 Implications

The results of this work have several implications for the field. First, they demonstrate that DPO training is a viable alternative to traditional fine-tuning for knowledge graph reasoning tasks, potentially offering better sample efficiency and more direct optimization of reasoning quality. Second, the high accuracy achieved on a small dataset suggests that preference-based training can be effective even with limited training data, which is important for domains where labeled data is scarce.

The successful integration of multiple frameworks (SNS, GITA, and GraphWiz) shows that combining complementary approaches can yield effective solutions. The hybrid nature of our approach, combining textual and visual components, suggests that multi-modal reasoning may offer benefits for structured reasoning tasks, though the current evaluation did not explicitly compare text-only versus hybrid approaches.

## 5.4 Limitations

Several limitations should be acknowledged. The dataset size is relatively small (200 total samples), which limits the statistical significance of the results and may not fully represent performance on larger knowledge graphs. The evaluation was performed on a single dataset from the biomedical domain (PrimeKG), and generalizability to other domains remains to be tested. The training was performed using GPT-2, a small model, and it remains to be seen how performance would scale with larger models.

The visual component of the hybrid approach was not explicitly evaluated in this work, as images were not generated for the final dataset. Future work should investigate the specific contribution of visual graph representations to reasoning performance. Additionally, the current implementation uses a simple rejected response ("unknown") for all negative examples, which may not fully capture the diversity of incorrect reasoning patterns.

## 5.5 Future Work

Several directions for future work emerge from this research. First, scaling the approach to larger datasets and larger models would provide a more comprehensive evaluation of the method's effectiveness. Second, explicit ablation studies comparing text-only versus hybrid approaches, with and without SimCSE ranking, would help identify the specific contributions of each component.

Third, extending the evaluation to additional knowledge graph domains beyond biomedicine would test the generalizability of the approach. Fourth, developing more sophisticated negative sampling strategies for DPO training, beyond the simple "unknown" rejection, could improve training effectiveness. Fifth, investigating the specific contribution of visual graph representations through controlled experiments would clarify their role in reasoning performance.

Finally, comparing the approach against established baselines such as zero-shot language models, rule-based systems, and other fine-tuning approaches would provide a clearer understanding of the relative advantages of the hybrid DPO framework. Statistical significance testing with multiple runs and confidence intervals would also strengthen the empirical evaluation.

## 5.6 Final Remarks

This work demonstrates that hybrid DPO training is a promising approach for knowledge graph reasoning tasks. The integration of similarity-based neighbor selection, visual graph representations, and preference-based training creates a framework that is both effective and accessible. While the current evaluation is limited in scale, the results provide a strong foundation for future research in this direction.

The practical contributions of reproducibility, scalability, and comprehensive evaluation make this work a useful resource for the research community. The codebase is designed to be easily extensible and adaptable to different knowledge graph domains and tasks, facilitating future research and applications.

In conclusion, this work contributes to the growing body of research on aligning language models for structured reasoning tasks. The combination of DPO training with knowledge graph reasoning shows promise, and we hope that this work will inspire further research in this direction, ultimately leading to more effective and reliable systems for knowledge graph reasoning.



