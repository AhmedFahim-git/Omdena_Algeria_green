import numpy as np
import numpy.typing as npt
import pandas as pd

# Credits to https://gist.github.com/gyosit/abeab4e595d7ddcd65b55c1270d240c8
# Reference
# Jones(1999) "Reduced state-variable tomato growth model"
# Jones(1991) "A dynamix tomato growth and yield model(TOMGRO)"
# Dimokas(2009) "Calibration and validation of a biological model..."
# Heuvelink(1994) "Dry-matter partitioning in a tomato crop:Comparison of two simulation models"


class TOMGRO:
    def __init__(self, dens=1.95, N=2.17, LAI=0.0028, Wf=0.0, W=0.175, Wm=0.0):
        # dens is plants per square metre: 1.4 if reference and 1.95 if reference
        # Initial data
        self.init_val_dict = {
            "dens": dens,
            "N": N,
            "LAI": LAI,
            "Wf": Wf,
            "W": W,
            "Wm": Wm,
        }
        self.dens = dens
        self.N = [N]
        self.LAI = [LAI]
        self.Wf = [Wf]
        self.W = [W]
        self.Wm = [Wm]

        # Growth per day
        self.dN: list[float] = []
        self.dLAI: list[float] = []
        self.dW: list[float] = []
        self.dWf: list[float] = []
        self.dWm: list[float] = []

    def reset_init_values(self):
        self.dens = self.init_val_dict["dens"]
        self.N = [self.init_val_dict["N"]]
        self.LAI = [self.init_val_dict["LAI"]]
        self.Wf = [self.init_val_dict["Wf"]]
        self.W = [self.init_val_dict["W"]]
        self.Wm = [self.init_val_dict["Wm"]]

        # Growth per day
        self.dN: list[float] = []
        self.dLAI: list[float] = []
        self.dW: list[float] = []
        self.dWf: list[float] = []
        self.dWm: list[float] = []

    ##########
    # dN/dt : The rate of node development
    def __calculate_fN(self, T_: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # T is the mean hourly temperature in degree Celsius
        # Shamshiri (2016)
        V0 = 0.25 + 0.025 * T_
        V1 = 2.5 - 0.05 * T_
        return np.minimum(1, V0, V1)

    def __calculate_dNdt(self, fN_: npt.NDArray[np.float64]) -> float:
        # Computed hourly then integrated to daily
        Nm = 0.55  # P Jones(1991)
        # ("dNdt:", Nm * fN_)
        return (Nm * fN_).sum()

    ##########
    # d(LAI)/dt : The rate of LAI(Leaf Area Index) development

    def __calculate_lambda(self, Td_: float) -> float:
        # Td average daily temperature
        # Lambda is a temperature function to reduce rate of leaf area expansion. It is an unitless (0 to 1 function)
        return 1.0

    def __calculate_dLAIdt(
        self, LAI_: float, N_: float, lambda_: float, dNdt_: float
    ) -> float:

        delta = 0.030  # P Maximum leaf area expansion per node, coefficient in expolinear equation; Jones(1999)
        beta = 0.169  # P Coefficient in expolinear equation; Jones(1999)
        Nb = 16.0  # P Coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis; Jones(1999)
        LAImax = 4.0  # P Jones(1999)

        if LAI_ > LAImax:
            return 0

        a: float = np.exp(beta * (N_ - Nb))
        return self.dens * delta * lambda_ * a * dNdt_ / (1 + a)

    ##########
    # dWfdt : The rate of Fruit dry weight

    def __calculate_LFmax(
        self, CO2_: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # maximum leaf photosyntehstic rate;Jones(1991)
        # CO2[ppm] hourly
        tau = 0.0693  # P carbon dioxide use efficiency; Jones(1991)
        return tau * CO2_

    def __calculate_PGRED(self, T_: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # function to modify Pg under suboptimal daytime temperatures; Jones(1991)
        PGRED_: npt.NDArray[np.float64] = np.where(T_ > 35, 0, T_)
        PGRED_ = np.where(PGRED_ > 12, 1, PGRED_ / 12)
        return PGRED_

    def __calculate_Pg(
        self,
        LFmax_: npt.NDArray[np.float64],
        PGRED_: npt.NDArray[np.float64],
        PPFD_: npt.NDArray[np.float64],
        LAI_: float,
    ) -> float:
        D = 2.593  # P coefficient to convert Pg from CO2 to CH2O; Jones(1991)
        K = 0.58  # P light extinction coefficient; Jones(1991)
        m = 0.1  # P leaf light transmission coefficient; Jones(1991)
        Qe = 0.0645  # P leaf quantum efficiency; Jones(1991)

        # if(PPFD > 250):
        #  PPFD = 0
        a = D * LFmax_ * PGRED_ / K
        b: npt.NDArray[np.float64] = np.log(
            ((1 - m) * LFmax_ + Qe * K * PPFD_)
            / ((1 - m) * LFmax_ + Qe * K * PPFD_ * np.exp(-1 * K * LAI_))
        )
        return (a * b).sum()

    def __calculate_Rm(
        self, T_: npt.NDArray[np.float64], W_: float, Wm_: float
    ) -> float:
        # Jones(1999)
        # Hourly Data!
        Q10 = 1.4  # P Jones(1991)
        rm = 0.016  # P Jones(1999)
        return ((Q10 ** ((T_ - 20) / 10)) * rm * (W_ - Wm_)).sum()

    def __calculate_fR(self, N_: float) -> float:
        # root phenology-dependent fraction; Jones(1991)
        if N_ >= 30:
            return 0.07
        return -0.0046 * N_ + 0.2034

    def __calculate_GRnet(self, Pg_: float, Rm_: float, fR_: float) -> float:
        E = 0.717  # P convert efficiency; Dimokas(2009)
        # print("GRnet", Pg_, Rm_, fR_)
        return max(0, E * (Pg_ - Rm_) * (1 - fR_))

    # Might need extra check
    def __calculate_fF(self, Td_: float) -> float:
        # Td is average daily temperature
        # Jones(1991)
        if Td_ > 8 and Td_ <= 28:
            return 0.0017 * Td_ - 0.0147
        elif Td_ > 28:
            return 0.032
        else:
            return 0

    def __calculate_g(self, T_daytime_: float) -> float:
        T_CRIT = 22  # P mean daytime temperature above which fruits abortion start; Jones(1999) (for Avignon France)
        if T_daytime_ < T_CRIT:
            return 1.0
        return 1.0 - 0.154 * (T_daytime_ - T_CRIT)

    def __calculate_dWfdt(
        self, GRnet_: float, fF_: float, N_: float, g_: float
    ) -> float:
        NFF = 19  # P Nodes per plant when first fruit appears; Jones(1999) (for Avignon France)
        alpha_F = 0.95  # P Maximum partitioning of new growth to fruit; Jones(1999) (for Avignon France)
        v = 0.2  # P Transition coefficient between vegetative and full fruit growth; Jones(1999) (for Avignon France)
        # fF_ = 0.5  # P ORIGINAL
        if N_ <= NFF:
            return 0
        # print(GRnet_, fF_, 1 - np.exp(v*(NFF-N)), g_)
        return GRnet_ * alpha_F * fF_ * (1 - np.exp(v * (NFF - N_))) * g_

    ##########
    # dWdt : The rate of Aboveground dry weight

    def __calculate_dWdt(self, GRnet_: float, dWfdt_: float, dNdt_: float) -> float:
        p1 = 2.0  # P Loss of leaf dry weight per node after LAImax is reached; Jones(1999)
        Vmax = 8  # P Maximum increase in vegetative tissue dry weight growth per node; Jones(1999) (for Avignon France)
        # print(GRnet_, fF_, 1 - np.exp(v*(NFF-N)), g_)
        return max(
            (GRnet_ - p1 * Vmax * dNdt_), (dWfdt_ + (Vmax - p1) * self.dens * dNdt_)
        )

    ##########
    # dWmdt
    def __calculate_Df(self, Td_: float) -> float:
        # The rate of development or aging of fruit at temperature T; Jones(1991)
        if Td_ > 9 and Td_ <= 28:
            return 0.0017 * Td_ - 0.015
        elif Td_ > 28 and Td_ <= 35:
            return 0.032
        else:
            return 0

    # Might need extra check
    def __calculate_dWmdt(self, Df_: float, Wf_: float, Wm_: float, N_: float) -> float:
        NFF = 19  # P Jones(1999) (for Avignon France)
        kF = 5.0  # P Jones(1999)

        if N_ <= NFF + kF:
            return 0
        return Df_ * (Wf_ - Wm_)

    def calc_one_day(
        self,
        T_: npt.NDArray[np.float64],
        PPFD_out_: npt.NDArray[np.float64],
        PPFD_: npt.NDArray[np.float64],
        CO2_: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        Td_: float = T_.mean()
        T_daytime_: float = T_[(PPFD_out_ > 0)].mean()

        fN_ = self.__calculate_fN(T_=T_)
        dNdt_ = self.__calculate_dNdt(fN_=fN_)

        lambda_ = self.__calculate_lambda(Td_=Td_)
        dLAIdt_ = self.__calculate_dLAIdt(
            LAI_=self.LAI[-1], N_=self.N[-1], lambda_=lambda_, dNdt_=dNdt_
        )

        LFmax_ = self.__calculate_LFmax(CO2_=CO2_)
        PGRED_ = self.__calculate_PGRED(T_=T_)
        Pg_ = self.__calculate_Pg(
            LFmax_=LFmax_, PGRED_=PGRED_, PPFD_=PPFD_, LAI_=self.LAI[-1]
        )

        Rm_ = self.__calculate_Rm(T_=T_, W_=self.W[-1], Wm_=self.Wm[-1])
        fR_ = self.__calculate_fR(N_=self.N[-1])
        GRnet_ = self.__calculate_GRnet(Pg_=Pg_, Rm_=Rm_, fR_=fR_)
        fF_ = self.__calculate_fF(Td_=Td_)
        g_ = self.__calculate_g(T_daytime_=T_daytime_)

        dWfdt_ = self.__calculate_dWfdt(GRnet_=GRnet_, fF_=fF_, N_=self.N[-1], g_=g_)

        dWdt_ = self.__calculate_dWdt(GRnet_=GRnet_, dWfdt_=dWfdt_, dNdt_=dNdt_)

        Df_ = self.__calculate_Df(Td_=Td_)
        dWmdt_ = self.__calculate_dWmdt(
            Df_=Df_, Wf_=self.Wf[-1], Wm_=self.Wm[-1], N_=self.N[-1]
        )

        self.dN.append(dNdt_)
        self.dLAI.append(dLAIdt_)
        self.dWf.append(dWfdt_)
        self.dW.append(dWdt_)
        self.dWm.append(dWmdt_)

        new_N = self.N[-1] + dNdt_
        new_LAI = self.LAI[-1] + dLAIdt_
        new_Wf = self.Wf[-1] + dWfdt_
        new_W = self.W[-1] + dWdt_
        new_Wm = self.Wm[-1] + dWmdt_

        self.N.append(new_N)
        self.LAI.append(new_LAI)
        self.Wf.append(new_Wf)
        self.W.append(new_W)
        self.Wm.append(new_Wm)

        return {"N": new_N, "LAI": new_LAI, "Wf": new_Wf, "W": new_W, "Wm": new_Wm}

    def calc_from_hourly(self, input_df: pd.DataFrame) -> pd.DataFrame:
        def apply_for_each_day(my_df: pd.DataFrame) -> pd.Series:
            T_ = my_df["Tair"].to_numpy()
            PPFD_out_ = my_df["PARout"].to_numpy()
            PPFD_ = my_df["Tot_PAR"].to_numpy()
            CO2_ = my_df["CO2air"].to_numpy()
            return pd.Series(
                self.calc_one_day(T_=T_, PPFD_out_=PPFD_out_, PPFD_=PPFD_, CO2_=CO2_)
            )

        input_df = input_df.copy(deep=True)
        input_df["days_planted"] = input_df["days_planted"].astype(int)
        input_df["days_planted"] += 1
        res = pd.concat(
            [
                pd.DataFrame(
                    {
                        "days_planted": [0],
                        "N": self.N[:1],
                        "LAI": self.LAI[:1],
                        "Wf": self.Wf[:1],
                        "W": self.W[:1],
                        "Wm": self.Wm[:1],
                    }
                ),
                input_df.groupby("days_planted")
                .apply(apply_for_each_day)
                .reset_index(),
            ]
        ).reset_index(drop=True)
        res["datetime"] = pd.to_datetime("2019-12-16") + pd.to_timedelta(
            res["days_planted"], unit="day"
        )
        res = res.set_index("datetime", drop=True)
        self.reset_init_values()
        return res
