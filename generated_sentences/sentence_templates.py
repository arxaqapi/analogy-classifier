# Common capital cities template sentences
ccc_sentences = (
    ("They traveled to $", "They took a trip to $"),
    ("She arrived yesterday in $", "She just landed in $"),
    ("We just came back from $", "We arrived yesterday from $"),
    ("I took my vacations in $", "I flew to $ for these summer vacation"),
    (
        "My internship supervisor is born in $",
        "My internship supervisor comes from $"
    ),
    ("Tomorrow I'll be going to $", "I have to travel to $ tomorrow"),
    (
        "These pretended journeys to $ were rather cumbrous.",
        "These so-called trips to $ were rather painful."
    )
)

# city-in-state
cis_sentences = (
    ("They go down to $"),
    ("His sister went to $"),
    ("The president of the United States was born in $"),
    ("Citizens in $ are more likely to vote"),
    ("Citizens in $, upset over changes to a bridge, negotiated a deal in their favour"),
    ("$ labor economists think a shortage is probable"),
    ("Its bank in $ also reported an increased loss for the quarter"),
    ("Several people are protesting in front of a controversial construction site in $")
)

# currency
cur_sentences = (
    ("The economy in $ is great", "The $ appreciated due to the strong economy"),
    (
        "Financial markets in $ are performing well, attracting investors",
        "Brokers are betting on the $"
    ),
    ("What is the currency used in $?", "Which country uses the $?"),
    (
        "In $, the local bank portfolio plunged to a very low value",
        "The bank lost multiple millions of $"
    )
)

# family
fam_sentences = (
    # skip: his, her | he, she | prince, princess| policeman, policewoman
    # if word finish with 's' , modify is with are
    ("His $ could not be present at the annual family gathering"),
    ("One son sacrifices his own career so that his avaricious $ can succeed"),
    ("My birth certificate has been falsified by my $"),
    ("Her $ [is,are] getting older every day"),
    ("My friend's $ [is,are] sick"),
    ("Everybody at the wedding party is dancing except my $"),
    ("She is like her $"),
    ("Let me introduce to you my $"),
    ("While my $ [is,are] at the bank, I started cooking for everyone"),
    ("Let him take care of his $"),
    ("If you were not my $ there would be nothing I could reproach you with"),
    ("Her $ [is,are] very rich and stingy"),
    ("The $ [is,are] coming. Can you hear them?")    
)


# gram6-nationality-adjective template sentences | Egypt, Egyptian
gna_sentences = (
    # an if $ starts with [a,e,i,o,u]
    ("My friend comes from $", "I have [a,an] $ friend"),
    ("The culture in $ is very rich", "The $ culture is very rich"),
    ("The whole family comes from $", "This family is $"),
    ("The man from $ tapped his cheeck", "The $ man tapped his cheeck"),
    (
        "They serve food from $ in this new restaurant",
        "There's a new $ restaurant near my house."
    ),
    ("Emma's new boyfriend comes from $ .", "Emma has [a,an] $ boyfriend"),
    (
        "His parents came from $, even if he was born in Hong Kong.",
        "Although he was born in Hong Kong, both of his parents are $."
    ),
    ("The people in $ are warm and friendly", "$ people are warm and friendly."),
    (
        "Like most people from $ ,she loves football.",
        "Like most $ ,she loves football."
    ),
    (
        "Controversial bills are being passed in $",
        "The $ government is passing controversial bills"
    ),
    (
        "The health situation is only getting worse in $",
        "The $ government is unable to cope with the rapid deterioration of the health situation"
    ),
    ("He speaks a refined langage only spoken in $", "He spoke in that refined $"),
    (
        "A regiment of five hundred thousand men attacked $",
        "A total force of five hundred thousand men attacked the $ from different sides."
    )
)


# opposite
opp_sentences = (  # switch with opposite, get 2 sentences with relations r
    ("A man is on a rock high above some trees and is standing in an [comfortable] position"),
    ("A man is sitting and tables a [pleasant] discussion"),
    ("At our time of life it is [pleasant] to be making new acquaintances every day"),
    ("He had never met with more [pleasant] people in his life"),
    ("I was [aware] in wich direction he was going"),
    ("The old man is [aware] that we are interesting ourselves in his affairs"),
    ("It was obvious, now, that the whale had at length become [aware] of his pursuers."),
    ("A specific order is [acceptable] to an exchange"),
    ("They consider this offer to be [acceptable]"),
    ("These parties were [acceptable] to all"),
    ("With such an husband her misery was considered [certain]"),
    ("He spoke of it as a [certain] event, of which the time alone could be undecided"),
    ("He spoke of it as a certain event, of which the time alone could be [decided]"),
    ("I ask only a [comfortable] home"),
    ("He also felt relatively [comfortable]")
)

def count_sentence_templates():
    n_ccc = len(ccc_sentences) * 2
    n_cis = len(cis_sentences)
    n_cur = len(cur_sentences) * 2
    n_fam = len(fam_sentences)
    n_gna = len(gna_sentences) * 2
    n_app = len(opp_sentences)
    return n_ccc + n_cis + n_cur + n_fam + n_gna + n_app
