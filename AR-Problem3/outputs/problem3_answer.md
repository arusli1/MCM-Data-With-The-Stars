# Problem 3: Written Answer

## How much do pro dancers and celebrity characteristics impact how well a celebrity will do?

We modeled three outcomes using celebrity **age**, **industry**, and **pro partner**: (1) mean judge score over weeks 1–3, (2) mean estimated fan vote share over weeks 1–3, and (3) a **success score** (1 = winner, 0 = last place in the season).

**Magnitude of impact:**
- **Age** has a moderate negative effect: older celebrities tend to receive lower judge scores (r ≈ -0.35), lower fan share (r ≈ -0.28), and lower success. Younger contestants do better on average.
- **Industry** (e.g., Actor, Athlete, Singer) explains a modest share of variance (η² on the order of 0.01–0.06); some industries are associated with higher scores and success.
- **Pro partner** has the **largest** impact among the three: η² for judge score is about 0.11 and for fan share about 0.04. Certain pros (e.g., Derek Hough, Cheryl Burke, Julianne Hough) are consistently associated with higher contestant success even after controlling for age and industry (see residualized “pro boost” analysis in the EDA).

**How much is explained overall?**  
A linear model with age + industry + pro partner explains roughly **0.24** of the variance in judge scores, **0.13** in fan share, and **0.28** in success score. So these factors matter, but a large share of variance remains unexplained (talent, week-to-week performance, fan base size, etc.).

---

## Do they impact judges’ scores and fan votes in the same way?

**No.** The same factors do **not** affect judges and fans in the same way:

1. **Age:** The correlation of age with judge score (r ≈ -0.35) is **stronger** than with fan share (r ≈ -0.28). Judges tend to reward younger celebrities more than fans do (or fans are relatively more supportive of older contestants).

2. **Industry:** Industry effects (η²) are larger for judge scores than for fan share. Judges differentiate more by celebrity type than fans do.

3. **Pro partner:** The **biggest** discrepancy is for pro partner: η² for judges is about 0.11 vs about 0.04 for fans. So **pro partner matters more for judge scores** than for fan votes. Top pros are associated with higher judge scores; the association with fan share is weaker.

4. **Improvement over the season (W1→W3):** In the EDA, age predicts **fan** improvement (younger celebs gain more fan share over weeks 1–3) but does not significantly predict **judge** score improvement. So fans and judges respond differently to trajectory as well.

**Conclusion:** Pro dancers and celebrity characteristics (age, industry) all impact how well a celebrity does, with **pro partner** and **age** being especially important. These factors do **not** impact judges and fans in the same way: effects are generally **stronger on judge scores** than on fan votes, with pro partner showing the largest gap. Judges appear to weight technical/partner quality and youth more than fans do.
