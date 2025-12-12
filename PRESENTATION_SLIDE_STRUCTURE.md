# Visual Slide Structure Guide
## Recommended Layouts for Each Slide

---

## **Slide 1: Research Question & Motivation**

**Layout:**
```
┌─────────────────────────────────────┐
│  Title: Hybrid KG-LLM Reasoning   │
│  Subtitle: Multi-hop with DPO      │
├─────────────────────────────────────┤
│                                     │
│  [Problem Icon]                     │
│  The Challenge:                     │
│  • Over-squashing                   │
│  • Heterophily                      │
│  • Lack of alignment                │
│                                     │
│  [Solution Icon]                    │
│  Our Goal:                          │
│  • Hybrid DPO training              │
│  • Similarity-based selection       │
│  • Multi-modal reasoning            │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Simple icons for problem/solution
- Bullet points (not too many)
- Clean, readable font

---

## **Slide 2: High-Level Architecture**

**Layout:**
```
┌─────────────────────────────────────┐
│  Architecture Overview              │
├─────────────────────────────────────┤
│                                     │
│  [Stage 1]      [Stage 2]    [Stage 3]
│  Data Prep  →   DPO Training →  Eval
│                                     │
│  ┌──────────┐   ┌──────────┐  ┌──────┐
│  │ Extract  │   │ Fine-tune │  │ Link │
│  │ Triples  │   │ LLM       │  │ Pred │
│  │          │   │           │  │      │
│  │ Generate │   │ Learn     │  │ Multi│
│  │ Paths    │   │ Prefs     │  │ -hop │
│  │          │   │           │  │ QA   │
│  │ Visualize│   │           │  │      │
│  └──────────┘   └──────────┘  └──────┘
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Flow diagram with arrows
- Three boxes for three stages
- Simple icons or text in each box

---

## **Slide 3: Hybrid Dataset Construction**

**Layout:**
```
┌─────────────────────────────────────┐
│  Hybrid Dataset: What Makes It     │
│  "Hybrid"?                          │
├─────────────────────────────────────┤
│                                     │
│  1. Multi-Modal Prompts             │
│     ┌─────────────────────┐          │
│     │ Text: Query +      │          │
│     │       Triples      │          │
│     │                    │          │
│     │ Visual: Graph      │          │
│     │        Image       │          │
│     └─────────────────────┘          │
│                                     │
│  2. DPO Pairs                        │
│     Chosen ✓  vs  Rejected ✗        │
│                                     │
│  3. SimCSE Ranking (Optional)       │
│     Top-K + Threshold Filter        │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Example prompt box (text + image placeholder)
- DPO pair visualization (side-by-side)
- SimCSE ranking diagram

---

## **Slide 4: DPO Training** ⭐ **MOST IMPORTANT**

**Layout:**
```
┌─────────────────────────────────────┐
│  DPO Training: Core Innovation      │
├─────────────────────────────────────┤
│                                     │
│  Why DPO?                           │
│  • Learns from preferences          │
│  • Directly optimizes reasoning     │
│  • More sample-efficient            │
│                                     │
│  How It Works:                      │
│  ┌──────────┐                       │
│  │ Prompt   │ → [Model] → Chosen ✓  │
│  │ + Triples│                       │
│  │ + Image  │ → [Model] → Rejected✗ │
│  └──────────┘                       │
│                                     │
│  Training Config:                   │
│  • Models: GPT-2 to Mistral-7B     │
│  • LR: 5e-6, Batch: 1-4            │
│  • Epochs: 2-3                      │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Flow diagram showing prompt → model → chosen/rejected
- Training configuration table
- Emphasize this slide (larger font, more visual space)

---

## **Slide 5: Integration Framework**

**Layout:**
```
┌─────────────────────────────────────┐
│  Combining Three Approaches         │
├─────────────────────────────────────┤
│                                     │
│  [SNS]      [GITA]      [GraphWiz] │
│  SimCSE     Visual      DPO        │
│  Ranking    Graphs      Training   │
│     ↓          ↓            ↓       │
│     └──────────┴────────────┘      │
│              ↓                      │
│     Our Hybrid Framework            │
│                                     │
│  Novel Contributions:               │
│  • Dual-stage caching               │
│  • HPC-aware rendering              │
│  • DPO ablation tooling             │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Three boxes for three approaches
- Arrows converging to "Our Framework"
- List of novel contributions

---

## **Slide 6: Experimental Setup**

**Layout:**
```
┌─────────────────────────────────────┐
│  Experimental Setup                 │
├─────────────────────────────────────┤
│                                     │
│  Dataset:          Models:          │
│  • PrimeKG         • GPT-2 (124M)   │
│  • Biomedical KG   • Mistral-7B     │
│  • 80/10/10 split                  │
│                                     │
│  Tasks:            Metrics:         │
│  • Link Pred       • MRR            │
│  • Multi-hop QA    • Hits@10        │
│                    • Accuracy       │
│                    • F1             │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Two-column layout
- Clean table or bullet points
- Keep it simple and readable

---

## **Slide 7: Results & Findings**

**Layout:**
```
┌─────────────────────────────────────┐
│  Results & Findings                 │
├─────────────────────────────────────┤
│                                     │
│  Training Results:                  │
│  • Loss: 0.46 (final)              │
│  • Time: ~20s (GPT-2, small)       │
│  • Convergence: ✓                   │
│                                     │
│  [Training Loss Curve]              │
│  (if available)                     │
│                                     │
│  Key Findings:                      │
│  1. DPO training works              │
│  2. Hybrid approach helps           │
│  3. Efficient & scalable            │
│                                     │
│  Future: Scale up, baselines        │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Training loss curve (if you have one)
- Key findings as numbered list
- Future work mention

---

## **Slide 8: Comparison & Standing Out**

**Layout:**
```
┌─────────────────────────────────────┐
│  How We Stand Out                   │
├─────────────────────────────────────┤
│                                     │
│  vs. SNS:    vs. GITA:   vs. GraphWiz:
│  + DPO       + KG focus  + Visual   │
│  + KG        + DPO       + SimCSE    │
│  + Visual    + SimCSE    + Pipeline  │
│                                     │
│  Unique Contributions:              │
│  1. First hybrid DPO for KG         │
│  2. Integration framework           │
│  3. Reproducible pipeline           │
│  4. Ablation tooling                │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Comparison table (3 columns)
- List of unique contributions
- Use checkmarks or plus signs for additions

---

## **Slide 9: Takeaways & Impact**

**Layout:**
```
┌─────────────────────────────────────┐
│  Takeaways & Impact                 │
├─────────────────────────────────────┤
│                                     │
│  Key Takeaways:                     │
│  ✓ DPO aligns LLMs for KG reasoning │
│  ✓ Hybrid approach improves         │
│  ✓ SimCSE reduces noise             │
│  ✓ End-to-end pipeline              │
│                                     │
│  Impact:                            │
│  • Novel methodology                │
│  • Practical (standard hardware)    │
│  • Scalable                         │
│  • Reproducible                     │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Checkmarks for takeaways
- Bullet points for impact
- Keep it concise

---

## **Slide 10: Conclusion**

**Layout:**
```
┌─────────────────────────────────────┐
│  Conclusion                         │
├─────────────────────────────────────┤
│                                     │
│  Summary:                           │
│  • Hybrid DPO framework             │
│  • Combines SNS + GITA + GraphWiz   │
│  • Effective learning                │
│  • Reproducible pipeline             │
│                                     │
│  Future Work:                        │
│  • Scale to larger datasets          │
│  • Baseline comparisons              │
│  • Statistical testing               │
│  • Adaptive sampling                 │
│                                     │
│  Thank You!                         │
│  Questions?                         │
│                                     │
└─────────────────────────────────────┘
```

**Visual Elements:**
- Summary as bullet points
- Future work list
- "Thank You" and "Questions?" at bottom

---

## **General Design Guidelines**

### Color Scheme
- **Primary**: Professional blue or dark color
- **Accent**: Use sparingly for highlights
- **Background**: White or light gray
- **Text**: Dark (black or dark gray)

### Typography
- **Title**: 36-44pt, bold
- **Body**: 24-28pt
- **Bullet points**: 20-24pt
- **Font**: Sans-serif (Arial, Helvetica, Calibri)

### Visual Elements
- **Icons**: Simple, consistent style
- **Diagrams**: Clean lines, clear labels
- **Images**: High resolution, relevant
- **Charts**: Clear axes, readable labels

### Consistency
- Same color scheme throughout
- Same font family
- Same icon style
- Same layout structure

### Accessibility
- High contrast (text vs background)
- Large enough font sizes
- Clear, simple diagrams
- Not too much text per slide

---

## **Tools for Creating Slides**

### Recommended:
- **PowerPoint** (Microsoft)
- **Keynote** (Apple)
- **Google Slides** (Online)
- **LaTeX Beamer** (Academic style)

### For Diagrams:
- **Draw.io** (Free, online)
- **Lucidchart** (Professional)
- **PowerPoint/Keynote shapes** (Built-in)

### For Icons:
- **Flaticon** (Free icons)
- **Font Awesome** (Icon fonts)
- **Simple Unicode symbols** (✓, ✗, →, etc.)

---

## **Slide Template Checklist**

For each slide:
- [ ] Clear title
- [ ] Readable font size (24pt+)
- [ ] Not too much text (6-8 bullet points max)
- [ ] Visual elements (diagrams, icons, images)
- [ ] Consistent color scheme
- [ ] High contrast
- [ ] Proofread for typos
- [ ] Test on projector/screen

---

## **Example Visual Elements to Create**

1. **Architecture Diagram**: Three-stage pipeline with arrows
2. **Example Prompt**: Show text + image placeholder
3. **DPO Training Flow**: Prompt → Model → Chosen/Rejected
4. **Integration Diagram**: Three approaches converging
5. **Training Loss Curve**: If you have training data
6. **Comparison Table**: vs. SNS, GITA, GraphWiz
7. **Sample Graph Visualization**: From your dataset

---

## **Quick Tips**

- **Keep it simple**: Don't overcrowd slides
- **Use visuals**: Diagrams > text walls
- **Practice timing**: 1-2 minutes per slide
- **Emphasize key points**: Bold, larger font, color
- **Be consistent**: Same style throughout
- **Test readability**: View from back of room

