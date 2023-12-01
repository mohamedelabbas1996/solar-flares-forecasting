import datetime
import pandas as pd
import pickle

ACTIVE_REGIONS_WITH_POSITIVE_FLAREING_EVENTS = [
    8113,
    12473,
    12497,
    12497,
    12497,
    12497,
    12529,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12615,
    12615,
    10537,
    10537,
    10537,
    10537,
    10537,
    10537,
    10591,
    10597,
    10597,
    10599,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10663,
    10667,
    10687,
    10687,
    10687,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10715,
    10713,
    10715,
    10715,
    11675,
    11719,
    11719,
    11731,
    11731,
    11739,
    11739,
    11745,
    11745,
    11755,
    11745,
    11777,
    11787,
    11817,
    11865,
    11865,
    11865,
    11865,
    11861,
    11875,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11877,
    11891,
    11897,
    11897,
    11899,
    11893,
    11893,
    11909,
    11947,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11977,
    11991,
    11989,
    11991,
    11991,
    12011,
    12017,
    12017,
    12017,
    12017,
    12027,
    12035,
    12035,
    12051,
    12051,
    12055,
    12065,
    12077,
    12087,
    12087,
    12087,
    12087,
    12087,
    12087,
    12085,
    12087,
    12087,
    12089,
    12087,
    12085,
    12087,
    12085,
    12087,
    12113,
    12113,
    12127,
    12149,
    12149,
    12151,
    12155,
    12157,
    12157,
    12169,
    12173,
    12173,
    12173,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12209,
    12209,
    12209,
    12241,
    12241,
    12241,
    12241,
    12241,
    12249,
    11401,
    11401,
    11401,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11445,
    11461,
    11471,
    11513,
    11513,
    11513,
    11513,
    11513,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11519,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11521,
    11563,
    11583,
    11611,
    11613,
    11613,
    11613,
    11613,
    11613,
    8113,
    12473,
    12497,
    12497,
    12497,
    12497,
    12529,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12615,
    12615,
    10537,
    10537,
    10537,
    10537,
    10537,
    10537,
    10591,
    10597,
    10597,
    10599,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10663,
    10667,
    10687,
    10687,
    10687,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10715,
    10713,
    10715,
    10715,
    11675,
    11719,
    11719,
    11731,
    11731,
    11739,
    11739,
    11745,
    11745,
    11755,
    11745,
    11777,
    11787,
    11817,
    11865,
    11865,
    11865,
    11865,
    11861,
    11875,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11877,
    11891,
    11897,
    11897,
    11899,
    11893,
    11893,
    11909,
    11947,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11977,
    11991,
    11989,
    11991,
    11991,
    12011,
    12017,
    12017,
    12017,
    12017,
    12027,
    12035,
    12035,
    12051,
    12051,
    12055,
    12065,
    12077,
    12087,
    12087,
    12087,
    12087,
    12087,
    12087,
    12085,
    12087,
    12087,
    12089,
    12087,
    12085,
    12087,
    12085,
    12087,
    12113,
    12113,
    12127,
    12149,
    12149,
    12151,
    12155,
    12157,
    12157,
    12169,
    12173,
    12173,
    12173,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12209,
    12209,
    12209,
    12241,
    12241,
    12241,
    12241,
    12241,
    12249,
    11401,
    11401,
    11401,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11445,
    11461,
    11471,
    11513,
    11513,
    11513,
    11513,
    11513,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11519,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11521,
    11563,
    11583,
    11611,
    11613,
    11613,
    11613,
    11613,
    11613,
    8113,
    12473,
    12497,
    12497,
    12497,
    12497,
    12529,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12567,
    12615,
    12615,
    10537,
    10537,
    10537,
    10537,
    10537,
    10537,
    10591,
    10597,
    10597,
    10599,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10649,
    10663,
    10667,
    10687,
    10687,
    10687,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10691,
    10715,
    10713,
    10715,
    10715,
    11675,
    11719,
    11719,
    11731,
    11731,
    11739,
    11739,
    11745,
    11745,
    11755,
    11745,
    11777,
    11787,
    11817,
    11865,
    11865,
    11865,
    11865,
    11861,
    11875,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11875,
    11875,
    11877,
    11875,
    11875,
    11875,
    11877,
    11891,
    11897,
    11897,
    11899,
    11893,
    11893,
    11909,
    11947,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11967,
    11977,
    11991,
    11989,
    11991,
    11991,
    12011,
    12017,
    12017,
    12017,
    12017,
    12027,
    12035,
    12035,
    12051,
    12051,
    12055,
    12065,
    12077,
    12087,
    12087,
    12087,
    12087,
    12087,
    12087,
    12085,
    12087,
    12087,
    12089,
    12087,
    12085,
    12087,
    12085,
    12087,
    12113,
    12113,
    12127,
    12149,
    12149,
    12151,
    12155,
    12157,
    12157,
    12169,
    12173,
    12173,
    12173,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12205,
    12209,
    12209,
    12209,
    12241,
    12241,
    12241,
    12241,
    12241,
    12249,
    11401,
    11401,
    11401,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11429,
    11445,
    11461,
    11471,
    11513,
    11513,
    11513,
    11513,
    11513,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11513,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11519,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11515,
    11521,
    11563,
    11583,
    11611,
    11613,
    11613,
    11613,
    11613,
    11613,
    10865,
    10865,
    10875,
    10875,
    11041,
    11041,
    11041,
    11041,
    11041,
    11041,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11069,
    11081,
    11079,
    11093,
    1121,
    11121,
    11121,
    11121,
    7999,
    8869,
    8869,
    8889,
    8939,
    8925,
    8971,
    8977,
    8993,
    9031,
    9031,
    9041,
    9071,
    9071,
    9077,
    9077,
    9077,
    9077,
    9085,
    9087,
    9077,
    9087,
    9085,
    9087,
    9143,
    9151,
    9165,
    9165,
    9165,
    9213,
    9213,
    9221,
    9231,
    9235,
    9267,
    9289,
    9289,
    9289,
    10251,
    10251,
    10337,
    10337,
    10345,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10397,
    10397,
    10397,
    10421,
    10431,
    10431,
    10471,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    8179,
    8185,
    8195,
    8253,
    8293,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8323,
    8375,
    8375,
    8409,
    8415,
    8421,
    8419,
    8421,
    8421,
    12665,
    12665,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12685,
    10715,
    10719,
    10719,
    10759,
    10759,
    10759,
    10763,
    10763,
    10763,
    10763,
    10767,
    10775,
    10803,
    10803,
    8457,
    8471,
    8487,
    8487,
    8487,
    8485,
    8485,
    8525,
    8541,
    8583,
    8611,
    8603,
    8611,
    8645,
    8639,
    8649,
    8651,
    8649,
    8647,
    8731,
    8739,
    8749,
    8759,
    8759,
    8753,
    8759,
    8763,
    8771,
    8771,
    8771,
    8771,
    8771,
    8807,
    12253,
    12253,
    12257,
    12257,
    12257,
    12277,
    12277,
    12277,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12321,
    12325,
    12339,
    12339,
    12335,
    12335,
    12339,
    12339,
    12367,
    12365,
    12371,
    12371,
    12371,
    12371,
    12371,
    12367,
    12367,
    12371,
    12371,
    12381,
    12381,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12415,
    12415,
    12423,
    12423,
    12423,
    12423,
    12437,
    12437,
    12443,
    12445,
    12445,
    12443,
    12449,
    12473,
    12473,
    12473,
    12473,
    11149,
    11153,
    11161,
    11163,
    11165,
    11165,
    11165,
    11171,
    11165,
    11165,
    11165,
    11169,
    11169,
    11195,
    11195,
    11261,
    11261,
    11261,
    11263,
    11261,
    11261,
    11263,
    11263,
    11263,
    11283,
    1283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11301,
    11295,
    11295,
    11295,
    11303,
    11303,
    11303,
    11303,
    11305,
    11305,
    11305,
    11319,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11387,
    11387,
    11387,
    11389,
    11389,
    11389,
    10989,
    9767,
    9775,
    9775,
    9775,
    9773,
    9775,
    9775,
    9811,
    9871,
    9871,
    9885,
    9885,
    9899,
    9893,
    9893,
    9893,
    9961,
    9973,
    9973,
    9997,
    10017,
    10017,
    10017,
    10017,
    10017,
    10039,
    10039,
    10039,
    10039,
    10039,
    10039,
    10063,
    10057,
    10069,
    10061,
    10067,
    10061,
    10069,
    10069,
    10083,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10085,
    10069,
    10069,
    10069,
    10087,
    10085,
    10069,
    10069,
    10087,
    10083,
    10069,
    10087,
    10083,
    10083,
    10083,
    10083,
    10095,
    10095,
    10105,
    10105,
    10105,
    10105,
    10105,
    10137,
    10137,
    10137,
    10137,
    10139,
    10137,
    10139,
    10139,
    10139,
    10139,
    10159,
    10175,
    10177,
    10213,
    10225,
    10227,
    10229,
    10223,
    9313,
    9313,
    9313,
    9311,
    9373,
    9393,
    9401,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9415,
    9393,
    9415,
    9401,
    9415,
    9415,
    9415,
    9415,
    9415,
    9415,
    9433,
    9433,
    9433,
    9433,
    9445,
    9511,
    9511,
    9511,
    9557,
    9557,
    9591,
    9591,
    9591,
    9591,
    9591,
    9601,
    9601,
    9601,
    9601,
    9607,
    9631,
    9653,
    9671,
    9661,
    9661,
    9687,
    9715,
    9715,
    9727,
    9733,
    9727,
    9739,
    9751,
    10865,
    10865,
    10875,
    10875,
    11041,
    11041,
    11041,
    11041,
    11041,
    11041,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11069,
    11081,
    11079,
    11093,
    1121,
    11121,
    11121,
    11121,
    7999,
    8869,
    8869,
    8889,
    8939,
    8925,
    8971,
    8977,
    8993,
    9031,
    9031,
    9041,
    9071,
    9071,
    9077,
    9077,
    9077,
    9077,
    9085,
    9087,
    9077,
    9087,
    9085,
    9087,
    9143,
    9151,
    9165,
    9165,
    9165,
    9213,
    9213,
    9221,
    9231,
    9235,
    9267,
    9289,
    9289,
    9289,
    10251,
    10251,
    10337,
    10337,
    10345,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10397,
    10397,
    10397,
    10421,
    10431,
    10431,
    10471,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    8179,
    8185,
    8195,
    8253,
    8293,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8323,
    8375,
    8375,
    8409,
    8415,
    8421,
    8419,
    8421,
    8421,
    12665,
    12665,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12685,
    10715,
    10719,
    10719,
    10759,
    10759,
    10759,
    10763,
    10763,
    10763,
    10763,
    10767,
    10775,
    10803,
    10803,
    8457,
    8471,
    8487,
    8487,
    8487,
    8485,
    8485,
    8525,
    8541,
    8583,
    8611,
    8603,
    8611,
    8645,
    8639,
    8649,
    8651,
    8649,
    8647,
    8731,
    8739,
    8749,
    8759,
    8759,
    8753,
    8759,
    8763,
    8771,
    8771,
    8771,
    8771,
    8771,
    8807,
    12253,
    12253,
    12257,
    12257,
    12257,
    12277,
    12277,
    12277,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12321,
    12325,
    12339,
    12339,
    12335,
    12335,
    12339,
    12339,
    12367,
    12365,
    12371,
    12371,
    12371,
    12371,
    12371,
    12367,
    12367,
    12371,
    12371,
    12381,
    12381,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12415,
    12415,
    12423,
    12423,
    12423,
    12423,
    12437,
    12437,
    12443,
    12445,
    12445,
    12443,
    12449,
    12473,
    12473,
    12473,
    12473,
    11149,
    11153,
    11161,
    11163,
    11165,
    11165,
    11165,
    11171,
    11165,
    11165,
    11165,
    11169,
    11169,
    11195,
    11195,
    11261,
    11261,
    11261,
    11263,
    11261,
    11261,
    11263,
    11263,
    11263,
    11283,
    1283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11301,
    11295,
    11295,
    11295,
    11303,
    11303,
    11303,
    11303,
    11305,
    11305,
    11305,
    11319,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11387,
    11387,
    11387,
    11389,
    11389,
    11389,
    10989,
    9767,
    9775,
    9775,
    9775,
    9773,
    9775,
    9775,
    9811,
    9871,
    9871,
    9885,
    9885,
    9899,
    9893,
    9893,
    9893,
    9961,
    9973,
    9973,
    9997,
    10017,
    10017,
    10017,
    10017,
    10017,
    10039,
    10039,
    10039,
    10039,
    10039,
    10039,
    10063,
    10057,
    10069,
    10061,
    10067,
    10061,
    10069,
    10069,
    10083,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10085,
    10069,
    10069,
    10069,
    10087,
    10085,
    10069,
    10069,
    10087,
    10083,
    10069,
    10087,
    10083,
    10083,
    10083,
    10083,
    10095,
    10095,
    10105,
    10105,
    10105,
    10105,
    10105,
    10137,
    10137,
    10137,
    10137,
    10139,
    10137,
    10139,
    10139,
    10139,
    10139,
    10159,
    10175,
    10177,
    10213,
    10225,
    10227,
    10229,
    10223,
    9313,
    9313,
    9313,
    9311,
    9373,
    9393,
    9401,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9415,
    9393,
    9415,
    9401,
    9415,
    9415,
    9415,
    9415,
    9415,
    9415,
    9433,
    9433,
    9433,
    9433,
    9445,
    9511,
    9511,
    9511,
    9557,
    9557,
    9591,
    9591,
    9591,
    9591,
    9591,
    9601,
    9601,
    9601,
    9601,
    9607,
    9631,
    9653,
    9671,
    9661,
    9661,
    9687,
    9715,
    9715,
    9727,
    9733,
    9727,
    9739,
    9751,
    10865,
    10865,
    10875,
    10875,
    11041,
    11041,
    11041,
    11041,
    11041,
    11041,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11045,
    11069,
    11081,
    11079,
    11093,
    1121,
    11121,
    11121,
    11121,
    7999,
    8869,
    8869,
    8889,
    8939,
    8925,
    8971,
    8977,
    8993,
    9031,
    9031,
    9041,
    9071,
    9071,
    9077,
    9077,
    9077,
    9077,
    9085,
    9087,
    9077,
    9087,
    9085,
    9087,
    9143,
    9151,
    9165,
    9165,
    9165,
    9213,
    9213,
    9221,
    9231,
    9235,
    9267,
    9289,
    9289,
    9289,
    10251,
    10251,
    10337,
    10337,
    10345,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10365,
    10365,
    10365,
    10365,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10375,
    10397,
    10397,
    10397,
    10421,
    10431,
    10431,
    10471,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    10501,
    8179,
    8185,
    8195,
    8253,
    8293,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8307,
    8323,
    8375,
    8375,
    8409,
    8415,
    8421,
    8419,
    8421,
    8421,
    12665,
    12665,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12673,
    12685,
    10715,
    10719,
    10719,
    10759,
    10759,
    10759,
    10763,
    10763,
    10763,
    10763,
    10767,
    10775,
    10803,
    10803,
    8457,
    8471,
    8487,
    8487,
    8487,
    8485,
    8485,
    8525,
    8541,
    8583,
    8611,
    8603,
    8611,
    8645,
    8639,
    8649,
    8651,
    8649,
    8647,
    8731,
    8739,
    8749,
    8759,
    8759,
    8753,
    8759,
    8763,
    8771,
    8771,
    8771,
    8771,
    8771,
    8807,
    12253,
    12253,
    12257,
    12257,
    12257,
    12277,
    12277,
    12277,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12297,
    12321,
    12325,
    12339,
    12339,
    12335,
    12335,
    12339,
    12339,
    12367,
    12365,
    12371,
    12371,
    12371,
    12371,
    12371,
    12367,
    12367,
    12371,
    12371,
    12381,
    12381,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12403,
    12415,
    12415,
    12423,
    12423,
    12423,
    12423,
    12437,
    12437,
    12443,
    12445,
    12445,
    12443,
    12449,
    12473,
    12473,
    12473,
    12473,
    11149,
    11153,
    11161,
    11163,
    11165,
    11165,
    11165,
    11171,
    11165,
    11165,
    11165,
    11169,
    11169,
    11195,
    11195,
    11261,
    11261,
    11261,
    11263,
    11261,
    11261,
    11263,
    11263,
    11263,
    11283,
    1283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11283,
    11301,
    11295,
    11295,
    11295,
    11303,
    11303,
    11303,
    11303,
    11305,
    11305,
    11305,
    11319,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11339,
    11387,
    11387,
    11387,
    11389,
    11389,
    11389,
    10989,
    9767,
    9775,
    9775,
    9775,
    9773,
    9775,
    9775,
    9811,
    9871,
    9871,
    9885,
    9885,
    9899,
    9893,
    9893,
    9893,
    9961,
    9973,
    9973,
    9997,
    10017,
    10017,
    10017,
    10017,
    10017,
    10039,
    10039,
    10039,
    10039,
    10039,
    10039,
    10063,
    10057,
    10069,
    10061,
    10067,
    10061,
    10069,
    10069,
    10083,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10069,
    10085,
    10069,
    10069,
    10069,
    10087,
    10085,
    10069,
    10069,
    10087,
    10083,
    10069,
    10087,
    10083,
    10083,
    10083,
    10083,
    10095,
    10095,
    10105,
    10105,
    10105,
    10105,
    10105,
    10137,
    10137,
    10137,
    10137,
    10139,
    10137,
    10139,
    10139,
    10139,
    10139,
    10159,
    10175,
    10177,
    10213,
    10225,
    10227,
    10229,
    10223,
    9313,
    9313,
    9313,
    9311,
    9373,
    9393,
    9401,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9393,
    9415,
    9393,
    9415,
    9401,
    9415,
    9415,
    9415,
    9415,
    9415,
    9415,
    9433,
    9433,
    9433,
    9433,
    9445,
    9511,
    9511,
    9511,
    9557,
    9557,
    9591,
    9591,
    9591,
    9591,
    9591,
    9601,
    9601,
    9601,
    9601,
    9607,
    9631,
    9653,
    9671,
    9661,
    9661,
    9687,
    9715,
    9715,
    9727,
    9733,
    9727,
    9739,
    9751,
]


def read_df_from_csv(path):
    return pd.read_csv(path)


def write_df_to_csv(df, path):
    df.to_csv(path)


def read_df_from_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def write_df_to_pickle(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)


def map_noaa_to_harps_tarps(harps_noaa, tarps_noaa):
    df_harps = pd.read_csv(harps_noaa, sep=" ")
    df_tarps = pd.read_csv(tarps_noaa, sep=" ")
    noaa_tarps = {}
    noaa_harps = {}
    for idx, row in df_harps.iterrows():
        harp_num = row["HARPNUM"]
        noaa_num = row["NOAA_ARS"]
        for ar in noaa_num.split(","):
            noaa_harps[ar] = harp_num
    for idx, row in df_tarps.iterrows():
        tarp_num = row["TARPNUM"]
        noaa_num = row["NOAA_ARS"]
        for ar in noaa_num.split(","):
            noaa_tarps[ar] = harp_num
    return noaa_harps, noaa_tarps


def get_ars(file):
    lines = open(file).readlines()[1:]
    ars = []
    for line in lines:
        ars.append(line.split()[0])
    return ars


def convert2datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str[:-4], "%Y-%m-%d %H:%M:%S")


def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return datetime.datetime(year, month, day, hour, minute)
