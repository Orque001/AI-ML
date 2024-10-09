import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    Args:
    - people is a dictionary of people. The keys represent names, and 
      the values are dictionaries that contain mother and father keys. 
      You may assume that either mother and father are both blank (no 
      parental information in the data set), or mother and father will 
      both refer to other people in the people dictionary.
    - one_gene is a set of all people for whom we want to compute the 
      probability that they have one copy of the gene.
    - two_genes is a set of all people for whom we want to compute the 
      probability that they have two copies of the gene.
    - have_trait is a set of all people for whom we want to compute the 
      probability that they have the trait.

    The probability returned should be the probability that
    - everyone in set `one_gene` has one copy of the gene, and
    - everyone in set `two_genes` has two copies of the gene, and
    - everyone not in `one_gene` or `two_gene` does not have the gene, and
    - everyone in set `have_trait` has the trait, and
    - everyone not in set` have_trait` does not have the trait.

    For example, if the family consists of Harry, James, and Lily, then 
    calling this function where one_gene = {"Harry"}, two_genes = {"James"}, 
    and trait = {"Harry", "James"} should calculate the probability that 
    Lily has zero copies of the gene, Harry has one copy of the gene, 
    James has two copies of the gene, Harry exhibits the trait, James 
    exhibits the trait, and Lily does not exhibit the trait.

    For anyone with no parents listed in the data set, use the probability 
    distribution PROBS["gene"] to determine the probability that they have 
    a particular number of the gene.
    
    For anyone with parents in the data set, each parent will pass one of 
    their two genes on to their child randomly, and there is a 
    PROBS["mutation"] chance that it mutates (goes from being the gene to not 
    being the gene, or vice versa).
    
    Use the probability distribution PROBS["trait"] to compute the probability 
    that a person does or does not have a particular trait.
    """
    probability = 1

    # Go through each person in the dataset
    for person in people:
        genes = (2 if person in two_genes else
                 1 if person in one_gene else
                 0)
        trait = person in have_trait

        # If the person has parents in the dataset, calculate based on parents' genes
        if people[person]['mother'] is not None and people[person]['father'] is not None:
            # Get parents
            mother = people[person]['mother']
            father = people[person]['father']

            # Calculate the probability that each parent passes down the required genes
            mother_genes = (2 if mother in two_genes else
                            1 if mother in one_gene else
                            0)
            father_genes = (2 if father in two_genes else
                            1 if father in one_gene else
                            0)

            # Probabilities of passing on the gene
            mother_passing = 0.5 if mother_genes == 1 else (
                1 - PROBS["mutation"] if mother_genes == 2 else
                PROBS["mutation"]
            )
            father_passing = 0.5 if father_genes == 1 else (
                1 - PROBS["mutation"] if father_genes == 2 else
                PROBS["mutation"]
            )

            # Calculate the probability for the current person
            if genes == 2:
                probability *= mother_passing * father_passing
            elif genes == 1:
                probability *= (mother_passing * (1 - father_passing)) + ((1 - mother_passing) * father_passing)
            else:
                probability *= (1 - mother_passing) * (1 - father_passing)

        # If the person has no parents listed, use unconditional probability
        else:
            probability *= PROBS["gene"][genes]

        # Multiply by the probability of the person having/not having the trait
        probability *= PROBS["trait"][genes][trait]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.

    Args:
    - probabilities is a dictionary of people. Each person is mapped to a 
      "gene" distribution and a "trait" distribution.
    - one_gene is a set of people with one copy of the gene in the current 
      joint distribution.
    - two_genes is a set of people with two copies of the gene in the 
      current joint distribution.
    - have_trait is a set of people with the trait in the current joint 
      distribution.
    - p is the probability of the joint distribution.

    For each person person in `probabilities`, the function should update 
    the probabilities[person]["gene"] distribution and probabilities[person]["trait"]
    distribution by adding p to the appropriate value in each distribution. 
    All other values should be left unchanged.

    For example, if "Harry" were in both two_genes and in have_trait, then p would 
    be added to probabilities["Harry"]["gene"][2] and to 
    probabilities["Harry"]["trait"][True].

    The function should not return any value: it just needs to update the 
    probabilities dictionary.
    """
    # Loop through each person in the probabilities dictionary
    for person in probabilities:
        # Sets 2 to two_genes and 1 to one_gene
        genes = (2 if person in two_genes else
                 1 if person in one_gene else
                 0)
        # Update the probability for genes
        probabilities[person]['gene'][genes] += p
        # Update the probability of the person having a trait
        probabilities[person]['trait'][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    
    Args:
    - probabilities is a dictionary of people. Each person is mapped to a 
      "gene" distribution and a "trait" distribution.

    For both of the distributions for each person in probabilities, this 
    function should normalize that distribution so that the values in the 
    distribution sum to 1, and the relative values in the distribution are the same.

    For example, if probabilities["Harry"]["trait"][True] were equal to 0.1 and 
    probabilities["Harry"]["trait"][False] were equal to 0.3, then your function 
    should update the former value to be 0.25 and the latter value to be 0.75: the 
    numbers now sum to 1, and the latter value is still three times larger than 
    the former value.

    The function should not return any value: it just needs to update the 
    probabilities dictionary.
    """
    # Loop through each person in the probabilities dictionary
    for person in probabilities:
        # Loop through each field for the person
        for field in probabilities[person]:
            # Calculate the total probability for the probabilities
            total = sum(probabilities[person][field].values())
            # Loop through each value for the current field
            for value in probabilities[person][field]:
                # Normalize the probability dividing by the total probability
                probabilities[person][field][value] /= total


if __name__ == "__main__":
    main()
