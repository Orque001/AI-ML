from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # TODO
    Or(AKnight, AKnave),  # A can only be a knight or a knave, not both
    Not(And(AKnight, AKnave)),  # A cannot be both a knight and a knave
    Biconditional(AKnight, Not(AKnave)),  # If A is a knight, then A is not a knave
    Biconditional(AKnave, And(AKnight, AKnave))  # If A is a knave, then A lied about being a knight and a knave
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # TODO
    Or(AKnight, AKnave),  # A can only be a knight or a knave, not both.
    Or(BKnight, BKnave),  # B can only be a knight or a knave, not both.
    Implication(AKnight, And(Not(AKnave), Not(BKnave))),  # If A is a knight, then neither A nor B is a knave.
    Implication(AKnave, BKnight),  # If A is a knave, then B is a knight.
    Not(And(AKnight, AKnave)),  # It cannot be the case that A is both a knight and a knave.
    Not(And(BKnight, BKnave))  # It cannot be the case that B is both a knight and a knave.
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # TODO
    Or(AKnight, AKnave),  # A can only be a knight or a knave
    Or(BKnight, BKnave),  # B can only be a knight or a knave
    Not(And(AKnight, AKnave)),  # A cannot be both
    Not(And(BKnight, BKnave)),  # B cannot be both
    Or(
        And(AKnight, Not(BKnight)),  # If A is a knight, B is not a knight
        And(AKnave, BKnight)  # If A is a knave, B is a knight
    ),
    Or(
        And(AKnight, BKnight),  # If A is a knight, then B is also a knight
        And(AKnave, BKnave)  # If A is a knave, then B is also a knave
    ),
    Biconditional(AKnight, BKnave),  # A is a knight iff B is a knave
    Biconditional(BKnight, AKnave)  # B is a knight iff A is a knave
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # TODO
    Or(AKnight, AKnave),  # A can only be a knight or a knave
    Or(BKnight, BKnave),  # B can only be a knight or a knave
    Or(CKnight, CKnave),  # C can only be a knight or a knave
    Not(And(AKnight, AKnave)),  # A cannot be both
    Not(And(BKnight, BKnave)),  # B cannot be both
    Not(And(CKnight, CKnave)),  # C cannot be both
    Implication(BKnight, AKnave),  # If B is a knight, then A is a knave
    Implication(BKnight, CKnave),  # If B is a knight, then C is a knave
    Implication(CKnight, AKnight),  # If C is a knight, then A is a knight
    Implication(AKnave, Not(Implication(BKnight, AKnave))),  # If A is a knave, then B did not tell the truth when saying A is a knave
    Implication(CKnave, Not(CKnight))  # If C is a knave, C lied about A being a knight
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
