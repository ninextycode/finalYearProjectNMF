import requests
import sympy as sym
from theory.transform import expanded_expression_to_list
from itertools import chain
import json
import re
import numpy as np


class Interval:
    def __init__(self, min=-np.inf, max=np.inf,
                 min_included=False,
                 max_included=False):
        self.min = min
        self.max = max
        self.min_included = min_included
        self.max_included = max_included

    def get_middle_value(self):
        if self.min > self.max:
            raise Exception("{} is empty".format(self))
        v = (self.min + self.max) / 2
        return v

    def get_random_value(self):
        if self.min > self.max:
            raise Exception("{} is empty".format(self))
        v = self.min + (self.max - self.min) * np.random.rand()
        while v not in self:
            v = self.min + (self.max - self.min) * np.random.rand()
        return v

    def __contains__(self, item):
        left_border_check = self.min <= item if self.min_included else self.min < item
        right_border_check = self.max >= item if self.max_included else self.max > item
        return left_border_check and right_border_check

    def __str__(self):
        template = "{0}{1},{2}{3}"
        s = template.format(
            "[" if self.min_included else "(",
            self.min,
            self.max,
            "]" if self.max_included else ")"
        )
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def interval_from__string(s):
        wolfram_re = r"Inequality[(?P<lower_bound>\d+), " +\
                     "(?P<lower_decision>(LessEqual|Less)), " +\
                     "(?P<var>\d+), " +\
                     "(?P<upper_decision>(LessEqual|Less)), " +\
                     "(?P<upper_bound>\d+)]"
        print(wolfram_re)
        wolfram_match = re.match(wolfram_re, s)

        if wolfram_match:
            return wolfram_match.group("var"), \
                   Interval(
                        wolfram_match.group("lower_bound"),
                        wolfram_match.group("upper_bound"),
                        wolfram_match.group("lower_decision") == "LessEqual",
                        wolfram_match.group("upper_decision") == "LessEqual"
                    )

        inequality_re = r"(?P<var>\d+) (?P<sign>(<=|<|>|>=)) (?P<val>\d+)"
        inequality_match = re.match(wolfram_re, s)

        if inequality_match:
            sign = inequality_match.group("sign")
            if sign == "<=":
                return inequality_match.group("var"), \
                        Interval(-np.inf, float(inequality_match.group("val")), False, True)

            if sign == "<":
                return inequality_match.group("var"), \
                        Interval(-np.inf, float(inequality_match.group("val")), False, False)

            if sign == ">=":
                return inequality_match.group("var"), \
                        Interval(float(inequality_match.group("val")), np.inf, True, False)

            if sign == ">":
                return inequality_match.group("var"), \
                        Interval(float(inequality_match.group("val")), np.inf, False, False)
        raise Exception("Fail to parse string {}".format(s))


class Solver:
    def __init__(self, key_path):
        self.appid = "A3P4K4-77R44A258E"

    def solve_eq_0(self, eq_string):
        expanded_formula = expanded_expression_to_list(sym.expand(eq_string))
        request_str = self.sympy_formula_to_reuest_string(expanded_formula)
        self.send_request(request_str)

    def sympy_formula_to_reuest_string(self, expanded_formula):
        print(expanded_formula)
        request_str_template = "Plus[{}]==0,{}"

        var_set = sorted(set(chain.from_iterable([t[1] for t in expanded_formula])))
        var_range_str = ",".join(["0<={}<=1".format(v) for v in var_set])
        sum_terms_str = ",".join(["{}*{}".format(t[0], "*".join(t[1])) for t in expanded_formula])

        request_str = request_str_template.format(sum_terms_str, var_range_str)
        return request_str

    def send_request(self, request_str, vars):
        print(request_str)
        response = requests.get("http://api.wolframalpha.com/v2/query",
                                params={"appid": self.appid, "input": request_str,
                                        "output": "json", "format": "moutput"})
        pods = response.json()["queryresult"]["pods"]

        result = {v: Interval(0, 1, True, True) for v in vars}

        for pod in pods:
            if pod["title"] == "Solutions":
                result = pod["subpods"][0]["moutput"]
                for t in result:




s = Solver("")
s.solve_eq_0("x + y*z")


