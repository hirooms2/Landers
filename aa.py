from gritlm import GritLM

import torch

from gritlm_prompter import Prompter
from parser import parse_args
import torch.nn.functional as F


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


model = GritLM("GritLM/GritLM-7B", mode='embedding', torch_dtype="auto", num_items= 0, device="cpu")

args = parse_args()
prompter = Prompter(args)
query_instr, doc_instr = prompter.get_instruction()

queries = ["""System: Hi There! What types of movies do you like to watch? User: Hello! I'm more of an action movie or a good romance and mystery movie.""",

"""System: Hi There! What types of movies do you like to watch? User: Hello! I'm more of an action movie or a good romance and mystery movie.
System: I just saw the trailer for Knives Out when I went to see Joker and it looked like a good mix of action and mystery!
User: I seen that one too as I seen Joker about a month ago. I thought about asking my fiance about going and seeing it.
System: It looks like a good movie for people who like many different movies. It also has a great cast! I was surprised to see Chris Evans in the trailer!
User: Maybe with Chris Evans in it it'll be easier to convince my fiance to see it. Do you know who else is in the cast?
System: Daniel Craig and Jamie Lee Curtis are also in the cast. Daniel Craig does a lot of 007 so definitely a good hearthrob role to convince the misses lol!
User: I am the misses lol. But he loves the bond movies so that should be a good incentive for him to go see it. Do you have any other recommendations?
System: The new Star Wars comes out in less than a month, if you are into the franchise.
User: He is, I think he told me we're getting it when it comes out to add to our movie collection.
System: Well that is another great action movie. I also recommend the John Wick series.
User: I haven't seen any of that series. Could you tell me what the general plot is>.
System: John Wick is a former member of a gang, he was basically an assassin. He falls in love and quits the game, but then his wife dies, and someone comes in and kills his dog. He then goes on a revenge rampage against the people who broke into his house. I have yet to watch the 3rd one but the action scenes were really cool!
User: Oh I'd definitely would cry at the dogs death.
System: It is really sad! the dog was a last gift from his dying wife which makes it so much worse.
User: I couldn't even finish I am legend because of the dog dying. Anything with animal death makes me ball like a baby.
System: Marley & Me had me crying for a good half hour so I completely understand that!
User: I avoided that movie because someone told me he passed away. My fiance took me to see jurrasic world as our first date and I cried at the dinosuars dying.
System: I would definitely avoid that movie if animal deaths make you said. Oh that is so cute though!
User: Yeah, he had to calm me down for about an hour and bought me ice cream to apologize for it.""",

"""System: Hi! How's it going today.
User: Good how are you?
System: I'm good! Kind of lazy on this Sunday.
User: Still Saturday here! What movies did you see lately?
System: Oh! haha. I just re-watched Avengers End Game. It was great! I also just re-watched Terminator 2 and I LOVE that movie.
User: Which one did you like best?""",

"""'System: I just saw the trailer for Knives Out when I went to see Joker and it looked like a good mix of action and mystery!'"""]

documents = ['Knives Out (2019) director Rian Johnson', 'Knives Out (2019) writer Rian Johnson', 'Knives Out (2019) genre Comedy, Crime, Drama, Mystery, Thriller', 
             'Knives Out (2019) plot A detective investigates the death of a patriarch of an eccentric, combative family.', 'Knives Out (2019) actors Ana de Armas, Toni Collette, Chris Evans, Daniel Craig', 
             'Knives Out (2019) rating Internet Movie Database:7.6/10,Rotten Tomatoes:99%,Metacritic:84/100', 'Bandolero! (1968) writer James Lee Barrett (screenplay), Stanley Hough (story)', 
             'And Now the Screaming Starts! (1973) writer Roger Marshall (screenplay by), David Case (novel)', 'Endure (2010) actors Devon Sawa, Judd Nelson, Joey Lauren Adams, Clare Kramer', 
             'Batman: Mystery of the Batwoman (2003) writer Bob Kane (character created by: Batman), Alan Burnett (story), Michael Reaves',
             'Any Given Sunday (1999) genre Drama, Sport', 'Any Given Sunday (1999) director Oliver Stone', 'Any Given Sunday (1999) rating Internet Movie Database:6.9/10,Rotten Tomatoes:52%,Metacritic:52/100', 
              'Opposite Day (2009) director R. Michael Givens', 'Any Given Sunday (1999) actors Al Pacino, Cameron Diaz, Dennis Quaid, James Woods', 'Boomerang (1992) actors Eddie Murphy, Robin Givens, Halle Berry, David Alan Grier', 
              'Any Given Sunday (1999) writer Daniel Pyne (screen story), John Logan (screen story), John Logan (screenplay), Oliver Stone (screenplay)', 'Blankman (1994) actors Damon Wayans, David Alan Grier, Robin Givens, Christopher Lawford', 
              "The Witch's Curse (1962) writer Eddy H. Given (story by), Oreste Biancoli (screenplay), Piero Pierotti (screenplay)", "God's Not Dead 2 (2016) actors Jesse Metcalfe, Robin Givens, Melissa Joan Hart, Brad Heller",
              'Terminator 2: Judgment Day (1991) director James Cameron', 'Terminator Salvation (2009) director McG', 
              'Terminator: Dark Fate (2019) plot Sarah Connor and a hybrid cyborg human must protect a young girl from a newly modified liquid Terminator from the future.', 
              'Terminator: Dark Fate (2019) director Tim Miller', 'Terminator Genisys (2015) director Alan Taylor', 'Terminator 2: Judgment Day (1991) genre Action, Sci-Fi', 
              'Terminator Salvation (2009) genre Action, Sci-Fi', 'Terminator 3: Rise of the Machines (2003) director Jonathan Mostow', 'Terminator 2: Judgment Day (1991) writer James Cameron, William Wisher', 
              'Terminator Salvation (2009) writer John Brancato, Michael Ferris']




d_rep = model.encode(documents, instruction=gritlm_instruction(doc_instr))
q_rep = model.encode(queries, instruction=gritlm_instruction(query_instr))

d_rep = torch.from_numpy(d_rep)
q_rep = torch.from_numpy(q_rep)

cosine_similarity = F.cosine_similarity(q_rep.unsqueeze(1), d_rep.unsqueeze(0),dim=-1)
cos_sim = torch.where(torch.isnan(cosine_similarity), torch.full_like(cosine_similarity,0), cosine_similarity)
cos_sim = torch.softmax(cos_sim/0.02, dim=-1)
topk_sim_values, topk_sim_indices = torch.topk(cos_sim,k=30,dim=-1)

print()