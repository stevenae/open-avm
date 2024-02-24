# TL;DR

### I took Washington, D.C.'s open data and built an AI/ML model which produces home price predictions with <10% error. The results are available [on Google Sheets](url).

## Buying a House

Buying a house can be an intimidating process, not the least of which because of uncertainty around a home's value. Will I over-pay?

Many rely on price estimates from Zillow or Redfin, or traditional comps (comparable sales) provided by a real estate agent. Unfortunately, these estimates are usually inaccurate or imprecise.

Big company models have to be so many things to so many people, it is unsurprising that they do not meet the needs of all customers. Real estate agent comps, while intuitive, ignore the complex dynamics which actually drive home prices.

## Doing It Like the Big Guys

Large real estate investors have custom models which predict home prices and enable them to get good deals. The data to produce such models is expensive, luckily, the DC government provides much of the necessary data for free. Even more fortuitously, the data is of high-quality.

I took DC's data (available at [Open Data DC](https://opendata.dc.gov/)) and built a sophisticated model, which produces home price predictions with <10% median error (or, put another way, >90% median accuracy). Given that the data was free, it felt right to share the results for free, as well. 

The results are at agentoutcomes.com -- you can select any address available and immediately see our "nowcast" for home price. If there is interest, I'll expand to include comps.

The code is in this repo. 
