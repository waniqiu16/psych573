data {
  int<lower=0> J; // number of participants
  int yt[J]; // number of complex responses taught to baby
  int Nt[J]; // number of baby trials for participant
  int yc[J]; // number of complex responses taught to adult
  int Nc[J]; // number of adult trials for participant
}
parameters {
  vector<lower=0, upper=1>[J] thetac; // cluster-specific probabilities (adult)
  real<lower=0, upper=1> mu; // overall mean probability (adult)
  real<lower=0> kappa; // overall concentration (adult)
  vector[J] eta; // scaled difference in cluster-specific probabilities in probit scale
  real mu_delta; // overall difference between baby and adult on probit scale
  real<lower=0> tau_delta; // SD of dz in probit scale
}
transformed parameters {
  vector[J] thetat = Phi(inv_Phi(thetac) + mu_delta + eta * tau_delta);
}

model {
  yt ~ binomial(Nt, thetat); // each observation is binomial
  yc ~ binomial(Nc, thetac); // each observation is binomial
  thetac ~ beta_proportion(mu, kappa); // prior; Beta2 dist
  mu ~ beta(2, 2); // weak prior
  kappa ~ gamma(0.01, 0.01); // prior recommended by Kruschke
  eta ~ std_normal();
  mu_delta ~ std_normal(); // weak prior
  tau_delta ~ std_normal(); // weak prior
}

generated quantities {
  real mu_adult_basic; // adult-basic
  real mu_adult_diff_basic_complex;  // diff between adult-complex and adult-basic
  real mu_baby_complex; // baby-basic
  real mu_adult_chance;
  real mu_baby_basic;
  real mu_baby_diff_basic_complex;
  real mu_baby_chance;
  real mu_complex_babyadult;
  real mu_basic_babyadult;
  int ytrep[J];
  int ycrep[J];
  for (j in 1:J)
    ytrep[j] = binomial_rng(yt[J], mu);
  for (j in 1:J)
    ycrep[j] = binomial_rng(yc[J], mu);
  mu_adult_basic = 1-mu;
  mu_adult_diff_basic_complex = 2*mu - 1;
  mu_adult_chance = mu-0.5;
  mu_baby_complex = Phi(inv_Phi(mu) + mu_delta);
  mu_baby_basic = 1 - (Phi(inv_Phi(mu) + mu_delta));
  mu_baby_diff_basic_complex = Phi(inv_Phi(mu) + mu_delta) - (1 - (Phi(inv_Phi(mu) + mu_delta)));
  mu_baby_chance = Phi(inv_Phi(mu) + mu_delta) - 0.5;
  mu_complex_babyadult = mu - Phi(inv_Phi(mu) + mu_delta);
  mu_basic_babyadult = (1-mu) - (1 - (Phi(inv_Phi(mu) + mu_delta)));
}
