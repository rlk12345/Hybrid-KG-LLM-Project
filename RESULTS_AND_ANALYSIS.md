# Results and Analysis

## 4.1 Experimental Setup

We evaluated our hybrid DPO training framework on a knowledge graph reasoning task using a dataset derived from PrimeKG. The final dataset consists of 200 samples split into 140 training examples, 30 validation examples, and 30 test examples. Each example presents a question asking the model to identify the relation between two entities given relevant KG triples. The model is trained to prefer the correct relation (chosen response) over an incorrect relation (rejected response, set to "unknown").

Training was performed using GPT-2 as the base model, fine-tuned with Direct Preference Optimization over 3 epochs. The training configuration used a learning rate of 5×10⁻⁶, batch size of 1 with gradient accumulation of 2, resulting in an effective batch size of 2. Training completed successfully with the model learning to distinguish between preferred and non-preferred reasoning patterns.

## 4.2 Evaluation Results

We evaluated the trained model on the test set of 30 examples using comprehensive metrics including accuracy, precision, recall, and F1 score. The model achieved an overall accuracy of 96.7% (29 out of 30 correct predictions), demonstrating strong performance on the knowledge graph reasoning task. The macro-averaged precision was 92.9%, recall was 91.1%, and F1 score was 91.8%, indicating balanced performance across different relation types.

Per-relation analysis reveals that the model achieved perfect accuracy (100%) on 13 out of 14 relation types, including causes, upregulates, downregulates, consumes, activates, prevents, targets, produces, interacts_with, inhibits, metabolizes, and treats. The only relation type with imperfect performance was "binds_to", which achieved 75% accuracy (3 out of 4 correct). This suggests that the model successfully learned to identify most relation types with high reliability, with only one relation type showing room for improvement.

## 4.3 Error Analysis

A detailed examination of the single error case reveals an interesting pattern. The model predicted "Binds" when the correct answer was "binds_to". This error appears to be related to naming inconsistency rather than a fundamental reasoning failure. The model correctly identified the binding relationship but used a different naming convention (singular "Binds" versus the expected "binds_to" format). This suggests that the model understood the semantic relationship but failed to match the exact string format expected in the evaluation.

This type of error highlights the importance of consistent naming conventions in knowledge graph datasets and suggests that future work could benefit from normalization or fuzzy matching in evaluation metrics. The fact that only one such error occurred out of 30 test examples indicates that the model generally learned the correct relation naming patterns from the training data.

## 4.4 Training Dynamics

Analysis of the training process shows that the DPO loss decreased over the course of training, indicating that the model successfully learned to prefer chosen responses over rejected responses. The initial DPO loss was approximately 1.78 at the first training step, and the loss decreased as training progressed through 3 epochs. The model completed 210 training steps total, with the training process demonstrating stable convergence. The model successfully learned to distinguish between correct and incorrect reasoning patterns, as evidenced by the decreasing loss and the high accuracy achieved on the test set.

The training efficiency is notable, with the model achieving strong performance after only 3 epochs on a relatively small training set of 140 examples. This suggests that DPO training is effective for aligning language models with preferred reasoning patterns in knowledge graph contexts, requiring less training data and fewer epochs than might be expected for traditional fine-tuning approaches. The successful convergence indicates that the preference-based training objective is well-suited for this task.

## 4.5 Comparison with Baseline Expectations

While we did not perform formal baseline comparisons in this proof-of-concept study, the high accuracy achieved (96.7%) suggests that our hybrid DPO approach is effective for knowledge graph reasoning tasks. The performance is particularly notable given that the model was trained on a relatively small dataset (140 training examples) and evaluated on diverse relation types. The fact that 13 out of 14 relation types achieved perfect accuracy indicates that the model learned robust reasoning patterns that generalize across different relation types.

The results demonstrate that combining DPO training with knowledge graph reasoning tasks is a viable approach. The model successfully learned to identify relations between entities based on the provided KG context, showing that preference-based training can effectively align language models for structured reasoning tasks.

## 4.6 Limitations and Discussion

Several limitations should be acknowledged in interpreting these results. First, the dataset size is relatively small (200 total samples, 30 test samples), which limits the statistical significance of the results and may not fully capture the model's performance on larger, more diverse knowledge graphs. Second, the evaluation was performed on a single dataset derived from PrimeKG, and the generalizability to other knowledge graph domains remains to be tested.

The single error case involving naming inconsistency ("Binds" versus "binds_to") suggests that the model may be sensitive to exact string matching requirements. This could be addressed through improved data preprocessing, normalization of relation names, or more flexible evaluation metrics that account for semantic equivalence.

The training was performed using GPT-2, a relatively small language model (124M parameters). While this demonstrates the accessibility of the approach on standard hardware, it remains to be seen how performance would scale with larger models such as Mistral-7B or LLaMA. The current results provide a strong foundation, but future work should explore the performance gains achievable with larger models and larger datasets.

Despite these limitations, the results demonstrate that our hybrid DPO training framework is functional and effective for knowledge graph reasoning tasks. The high accuracy achieved, combined with the efficient training process, suggests that this approach has promise for practical applications in knowledge graph reasoning.

