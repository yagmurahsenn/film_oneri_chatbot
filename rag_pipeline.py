import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- A. GENEL AYARLAR ---
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "text-embedding-004" 
FILE_NAME = "movies_100.csv"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "movie_recommendations"


def create_and_save_data(filename):
    # Bu listeyi 100 filminizin tam verisiyle (süslü parantezli, virgüllü) doldurun!
   movie_data = [
    {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
    {"title": "The Godfather", "genre": "Crime, Drama", "year": 1972, "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "The Dark Knight", "genre": "Action, Crime, Drama", "year": 2008, "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
    {"title": "Pulp Fiction", "genre": "Crime, Drama", "year": 1994, "plot": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    {"title": "The Lord of the Rings: The Return of the King", "genre": "Adventure, Fantasy", "year": 2003, "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to give Frodo and Sam a chance to destroy the One Ring. A story of massive battles and final sacrifice."},
    {"title": "Forrest Gump", "genre": "Drama, Romance", "year": 1994, "plot": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart."},
    {"title": "Inception", "genre": "Action, Sci-Fi, Thriller", "year": 2010, "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
    {"title": "Fight Club", "genre": "Drama", "year": 1999, "plot": "An insomniac office worker looking for a way to change his life crosses paths with a devil-may-care soap maker and they form an underground fight club that evolves into something much, much more."},
    {"title": "The Matrix", "genre": "Action, Sci-Fi", "year": 1999, "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
    {"title": "Star Wars: Episode IV - A New Hope", "genre": "Action, Adventure, Fantasy", "year": 1977, "plot": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee, and two droids to save the galaxy from the Empire's world-destroying battle station, while also attempting to rescue a princess from the evil Darth Vader."},
    {"title": "Interstellar", "genre": "Sci-Fi, Adventure, Drama", "year": 2014, "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
    {"title": "Spirited Away", "genre": "Animation, Adventure, Family", "year": 2001, "plot": "During her family's move to the suburbs, a sullen ten-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts."},
    {"title": "Parasite", "genre": "Comedy, Drama, Thriller", "year": 2019, "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
    {"title": "Gladiator", "genre": "Action, Drama", "year": 2000, "plot": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."},
    {"title": "Whiplash", "genre": "Drama, Music", "year": 2014, "plot": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."},
    {"title": "The Prestige", "genre": "Drama, Mystery, Sci-Fi", "year": 2006, "plot": "After a tragic accident, two stage magicians engage in a bitter rivalry for years to create the ultimate illusion while sacrificing everything they have to achieve it."},
    {"title": "The Green Mile", "genre": "Crime, Drama, Fantasy", "year": 1999, "plot": "The lives of the staff on death row are affected by one of their charges: a black man accused of child murder and rape, yet possessing a mysterious gift."},
    {"title": "Léon: The Professional", "genre": "Action, Crime, Drama", "year": 1994, "plot": "Mathilda, a 12-year-old girl, is reluctantly taken in by Léon, a professional assassin, after her family is murdered. An unusual relationship develops as she learns the assassin's trade."},
    {"title": "The Usual Suspects", "genre": "Crime, Mystery, Thriller", "year": 1995, "plot": "A sole survivor tells of the events leading up to a disastrous fire on a boat, which leaves 27 dead and $91 million missing."},
    {"title": "Se7en", "genre": "Crime, Drama, Mystery", "year": 1995, "plot": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."},
    {"title": "Saving Private Ryan", "genre": "Drama, War", "year": 1998, "plot": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action."},
    {"title": "City of God", "genre": "Crime, Drama", "year": 2002, "plot": "Two boys growing up in a violent neighborhood of Rio de Janeiro take different paths: one becomes a photographer, the other a drug dealer."},
    {"title": "Terminator 2: Judgment Day", "genre": "Action, Sci-Fi", "year": 1991, "plot": "A cyborg is sent from the future to protect a young John Connor from a more advanced and powerful cyborg."},
    {"title": "Back to the Future", "genre": "Adventure, Comedy, Sci-Fi", "year": 1985, "plot": "Marty McFly, a high school student, is accidentally sent thirty years into the past in a time-traveling DeLorean invented by his friend, Dr. Emmett Brown."},
    {"title": "Alien", "genre": "Horror, Sci-Fi", "year": 1979, "plot": "The crew of a commercial space tug encounter a deadly alien lifeform after investigating a mysterious signal on a remote planet."},
    {"title": "The Silence of the Lambs", "genre": "Crime, Drama, Thriller", "year": 1991, "plot": "A young F.B.I. cadet must receive help from an incarcerated and manipulative cannibal killer to catch another serial killer, a madman who skins his victims."},
    {"title": "Reservoir Dogs", "genre": "Crime, Thriller", "year": 1992, "plot": "When a simple jewelry heist goes terribly wrong, the surviving criminals begin to suspect that one of them is a police informant."},
    {"title": "Apocalypse Now", "genre": "Drama, War", "year": 1979, "plot": "During the Vietnam War, a special operations officer is sent on a mission to assassinate a renegade Colonel who has set himself up as a god among a local tribe."},
    {"title": "Django Unchained", "genre": "Drama, Western", "year": 2012, "plot": "With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner."},
    {"title": "Inglourious Basterds", "genre": "Adventure, Drama, War", "year": 2009, "plot": "In Nazi-occupied France during World War II, a plan to assassinate the Nazi leaders by a group of Jewish U.S. soldiers coincides with a theater owner's plan for revenge."},
    {"title": "The Departed", "genre": "Crime, Drama, Thriller", "year": 2006, "plot": "An undercover state cop who has infiltrated an Irish gang and a mole in the police force working for the same mob struggle to identify one another before they are caught."},
    {"title": "Eternal Sunshine of the Spotless Mind", "genre": "Drama, Romance, Sci-Fi", "year": 2004, "plot": "When their relationship turns sour, a couple undergoes a procedure to have each other erased from their memories."},
    {"title": "Amélie", "genre": "Comedy, Romance", "year": 2001, "plot": "Amélie, an innocent and naive girl in Paris, decides to change the lives of those around her for the better."},
    {"title": "Goodfellas", "genre": "Biography, Crime, Drama", "year": 1990, "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his partners Jimmy Conway and Tommy DeVito."},
    {"title": "The Pianist", "genre": "Biography, Drama, Music", "year": 2002, "plot": "A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II."},
    {"title": "American History X", "genre": "Drama", "year": 1998, "plot": "A former neo-nazi skinhead tries to prevent his younger brother from following in his footsteps."},
    {"title": "The Lion King", "genre": "Animation, Adventure, Drama", "year": 1994, "plot": "Lion cub and future king Simba searches for his identity. His uncle, Scar, plots to usurp Mufasa's throne."},
    {"title": "The Grand Budapest Hotel", "genre": "Adventure, Comedy, Drama", "year": 2014, "plot": "The adventures of Gustave H, a legendary concierge at a famous European hotel, and Zero Moustafa, the lobby boy who becomes his most trusted friend."},
    {"title": "No Country for Old Men", "genre": "Crime, Drama, Thriller", "year": 2007, "plot": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and takes off with a case of money."},
    {"title": "A Clockwork Orange", "genre": "Crime, Drama, Sci-Fi", "year": 1971, "plot": "In a futuristic Britain, a gang leader is imprisoned and undergoes an experimental aversion therapy developed by the government in an attempt to curb criminal instincts."},
    {"title": "Shutter Island", "genre": "Mystery, Thriller", "year": 2010, "plot": "In 1954, a U.S. Marshal investigates the disappearance of a murderess who escaped from a hospital for the criminally insane."},
    {"title": "Donnie Darko", "genre": "Drama, Sci-Fi, Thriller", "year": 2001, "plot": "A troubled teenager is plagued by visions of a man in a large rabbit suit who manipulates him to commit a series of crimes."},
    {"title": "V for Vendetta", "genre": "Action, Drama, Sci-Fi", "year": 2005, "plot": "In a future British tyranny, a shadowy freedom fighter plots to overthrow the government with the help of a young woman."},
    {"title": "The Shining", "genre": "Drama, Horror", "year": 1980, "plot": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings from both past and future."},
    {"title": "Indiana Jones and the Raiders of the Lost Ark", "genre": "Action, Adventure", "year": 1981, "plot": "In 1936, archaeologist and adventurer Indiana Jones is hired by the U.S. government to find the Ark of the Covenant before the Nazis can obtain its awesome powers."},
    {"title": "The Thing", "genre": "Horror, Mystery, Sci-Fi", "year": 1982, "plot": "A research team in Antarctica is hunted by a shape-shifting alien that assumes the appearance of its victims."},
    {"title": "Drive", "genre": "Crime, Drama, Thriller", "year": 2011, "plot": "A mysterious Hollywood stuntman and mechanic moonlights as a getaway driver and finds himself in trouble when he helps out his neighbor."},
    {"title": "Oldboy", "genre": "Action, Drama, Mystery", "year": 2003, "plot": "After being kidnapped and imprisoned for fifteen years, a man is released, only to find that he must find his captor in five days."},
    {"title": "Pan's Labyrinth", "genre": "Drama, Fantasy, War", "year": 2006, "plot": "In 1944 Spain, a young girl, fascinated with fairy tales, is sent to live with her new stepfather, a cruel and sadistic army captain."},
    {"title": "WALL-E", "genre": "Animation, Adventure, Family", "year": 2008, "plot": "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind."},
    {"title": "Zodiac", "genre": "Crime, Drama, Mystery", "year": 2007, "plot": "In the late 1960s/early 1970s, a cartoonist becomes an amateur detective obsessed with tracking down the Zodiac Killer."},
    {"title": "Trainspotting", "genre": "Drama", "year": 1996, "plot": "The lives of a group of heroin addicts in Edinburgh, and their relationships with each other, are explored."},
    {"title": "Brazil", "genre": "Drama, Sci-Fi", "year": 1985, "plot": "A bureaucrat in a dystopic future attempts to correct a paperwork error, which leads him on a quest to find the girl of his dreams."},
    {"title": "Mad Max: Fury Road", "genre": "Action, Adventure, Sci-Fi", "year": 2015, "plot": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners and a drifter named Max."},
    {"title": "A Separation", "genre": "Drama", "year": 2011, "plot": "A married couple is faced with a difficult decision when the wife moves to another country and the husband stays behind to care for his father."},
    {"title": "Memories of Murder", "genre": "Crime, Drama, Thriller", "year": 2003, "plot": "In 1986, two detectives struggle with the case of a serial killer preying on women in a small South Korean province."},
    {"title": "Children of Men", "genre": "Action, Drama, Sci-Fi", "year": 2006, "plot": "In 2027, in a world where humanity has been infertile for eighteen years, a former activist agrees to transport a miraculously pregnant woman to a sanctuary at sea."},
    {"title": "The Secret in Their Eyes", "genre": "Drama, Mystery, Romance", "year": 2009, "plot": "A retired legal counselor writes a novel hoping to find closure for one of his past unresolved homicide cases and his unreciprocated love for his former boss."},
    {"title": "Fargo", "genre": "Crime, Drama, Thriller", "year": 1996, "plot": "Lying, an incompetent car salesman hires two thugs to kidnap his wife in an attempt to collect a ransom from his wealthy father-in-law."},
    {"title": "The Sixth Sense", "genre": "Drama, Mystery, Thriller", "year": 1999, "plot": "A boy who communicates with spirits who don't know they're dead seeks the help of a disheartened child psychologist."},
    {"title": "Snatch", "genre": "Comedy, Crime", "year": 2000, "plot": "Unscrupulous boxing promoters, bookies, and a Russian gangster collide over a stolen diamond."},
    {"title": "Before Sunrise", "genre": "Drama, Romance", "year": 1995, "plot": "A young man and woman meet on a train in Europe and spend a single night together in Vienna. Unfortunately, both know that this may be their only chance."},
    {"title": "The Truman Show", "genre": "Comedy, Drama, Sci-Fi", "year": 1998, "plot": "A man discovers his whole life is a reality TV show."},
    {"title": "Million Dollar Baby", "genre": "Drama, Sport", "year": 2004, "plot": "A determined woman works with a hardened boxing trainer to achieve her dream of becoming a professional boxer."},
    {"title": "Gravity", "genre": "Drama, Sci-Fi, Thriller", "year": 2013, "plot": "Two astronauts work together to survive after an accident leaves them adrift in space."},
    {"title": "Arrival", "genre": "Drama, Sci-Fi", "year": 2016, "plot": "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world."},
    {"title": "The Artist", "genre": "Comedy, Drama, Romance", "year": 2011, "plot": "A silent movie star meets a young dancer just as the age of talkies arrives."},
    {"title": "There Will Be Blood", "genre": "Drama", "year": 2007, "plot": "A story of family, religion, hatred, oil, and madness, focusing on a turn-of-the-century prospector in the early days of the oil industry."},
    {"title": "Requiem for a Dream", "genre": "Drama", "year": 2000, "plot": "The lives of four people are devastated by drug addiction."},
    {"title": "Prisoners", "genre": "Crime, Drama, Mystery", "year": 2013, "plot": "When two young girls go missing, the father of one girl takes matters into his own hands as the police pursue multiple leads."},
    {"title": "Inside Out", "genre": "Animation, Adventure, Comedy", "year": 2015, "plot": "After young Riley is uprooted to a new city, her emotions—Joy, Fear, Anger, Disgust and Sadness—conflict on how to navigate a new life."},
    {"title": "Coco", "genre": "Animation, Adventure, Family", "year": 2017, "plot": "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer."},
    {"title": "Blade Runner 2049", "genre": "Action, Drama, Sci-Fi", "year": 2017, "plot": "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard, who has been missing for thirty years."},
    {"title": "Spotlight", "genre": "Biography, Drama", "year": 2015, "plot": "The true story of how the Boston Globe's 'Spotlight' team uncovered the massive scandal of child molestation and cover-up within the local Catholic Archdiocese."},
    {"title": "The Social Network", "genre": "Biography, Drama", "year": 2010, "plot": "As Harvard student Mark Zuckerberg creates the social networking site that would become 'Facebook,' he is sued by the twins who claimed he stole their idea, and by the co-founder with whom he's fallen out."},
    {"title": "The King's Speech", "genre": "Biography, Drama, History", "year": 2010, "plot": "The story of King George VI, his impromptu accession to the throne, and the speech therapist who helped the monarch overcome his stammer."},
    {"title": "Argo", "genre": "Biography, Drama, Thriller", "year": 2012, "plot": "Acting under the cover of a Hollywood film crew, a CIA agent launches a dangerous operation to rescue six Americans in Tehran during the U.S. hostage crisis in Iran."},
    {"title": "Manchester by the Sea", "genre": "Drama", "year": 2016, "plot": "An uncle is forced to take care of his teenage nephew after the boy's father dies."},
    {"title": "La La Land", "genre": "Comedy, Drama, Music", "year": 2016, "plot": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."},
    {"title": "Roma", "genre": "Drama", "year": 2018, "plot": "A year in the life of a middle-class family's maid in Mexico City in the early 1970s."},
    {"title": "Birdman or (The Unexpected Virtue of Ignorance)", "genre": "Comedy, Drama", "year": 2014, "plot": "A washed-up actor, famous for portraying a superhero, tries to reclaim his artistic integrity with a Broadway play."},
    {"title": "Grand Torino", "genre": "Drama", "year": 2008, "plot": "Disgruntled Korean War veteran Walt Kowalski sets out to reform his teenage neighbor, who tried to steal Kowalski's prized 1972 Ford Gran Torino."},
    {"title": "Eternal Sunshine of the Spotless Mind", "genre": "Drama, Romance, Sci-Fi", "year": 2004, "plot": "When their relationship turns sour, a couple undergoes a procedure to have each other erased from their memories."},
    {"title": "Little Miss Sunshine", "genre": "Comedy, Drama", "year": 2006, "plot": "A family determined to get their daughter into the finals of a beauty pageant takes a cross-country trip in their VW bus."},
    {"title": "Before Sunset", "genre": "Drama, Romance", "year": 2004, "plot": "Nine years after Jesse and Celine first met, they encounter each other again on the French leg of Jesse's book tour."},
    {"title": "The Road", "genre": "Adventure, Drama, Sci-Fi", "year": 2009, "plot": "In a post-apocalyptic world, a father and his son walk a desolate road, trying to survive and keep the human spirit alive."},
    {"title": "Brokeback Mountain", "genre": "Drama, Romance", "year": 2005, "plot": "The story of a complicated and secret relationship between two cowboys over several years."},
    {"title": "Inglourious Basterds", "genre": "Adventure, Drama, War", "year": 2009, "plot": "In Nazi-occupied France during World War II, a plan to assassinate the Nazi leaders by a group of Jewish U.S. soldiers coincides with a theater owner's plan for revenge."},
    {"title": "Mystic River", "genre": "Crime, Drama, Mystery", "year": 2003, "plot": "The lives of three childhood friends are shattered when one of them has a family tragedy."},
    {"title": "Minority Report", "genre": "Action, Mystery, Sci-Fi", "year": 2002, "plot": "In a future where a special police unit can arrest murderers before they commit their crimes, an officer from that unit is accused of a future murder."},
    {"title": "Catch Me If You Can", "genre": "Biography, Crime, Drama", "year": 2002, "plot": "The true story of Frank Abagnale Jr. who successfully passed himself off as a pilot, a doctor, and a lawyer, all before his 21st birthday."},
    {"title": "Gangs of New York", "genre": "Crime, Drama", "year": 2002, "plot": "In 1862 New York City, a young Irish immigrant seeks revenge on the man who murdered his father."},
    {"title": "Million Dollar Baby", "genre": "Drama, Sport", "year": 2004, "plot": "A determined woman works with a hardened boxing trainer to achieve her dream of becoming a professional boxer."},
    {"title": "No Country for Old Men", "genre": "Crime, Drama, Thriller", "year": 2007, "plot": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and takes off with a case of money."},
    {"title": "Blade Runner", "genre": "Action, Sci-Fi, Thriller", "year": 1982, "plot": "A blade runner must pursue and terminate four replicants who stole a ship and have returned to Earth to find their creator."},
    {"title": "One Flew Over the Cuckoo's Nest", "genre": "Drama", "year": 1975, "plot": "A criminal pleads insanity and is admitted to a mental institution, where he rallies the other patients against the oppressive nurse."},
    {"title": "2001: A Space Odyssey", "genre": "Adventure, Sci-Fi", "year": 1968, "plot": "Humanity finds a mysterious, obviously artificial, monolith buried beneath the Lunar surface and, with the intelligent computer H.A.L. 9000, sets off on a quest."},
    {"title": "Psycho", "genre": "Horror, Mystery, Thriller", "year": 1960, "plot": "A Phoenix secretary embezzles $40,000 from her employer's client, goes on the run and checks into a remote motel run by a young man under the domination of his mother."},
    {"title": "Sunset Blvd.", "genre": "Drama, Film-Noir", "year": 1950, "plot": "A hack writer is taken in by an aging silent film star who dreams of a comeback."},
    {"title": "Casablanca", "genre": "Drama, Romance, War", "year": 1942, "plot": "A cynical American expatriate struggles to decide whether or not he should help his former lover and her husband escape French Morocco during World War II."},
    {"title": "Citizen Kane", "genre": "Drama, Mystery", "year": 1941, "plot": "Following the death of publishing tycoon Charles Foster Kane, reporters scramble to decipher the meaning of his final word: 'Rosebud.'"},
    {"title": "Modern Times", "genre": "Comedy, Drama, Romance", "year": 1936, "plot": "The Tramp struggles to live in modern industrial society with the help of a young homeless woman."},
    {"title": "The General", "genre": "Action, Adventure, Comedy", "year": 1897, "plot": "When Union spies steal an engineer's beloved locomotive, he pursues them doggedly and alone across enemy lines."},
    {"title": "Lawrence of Arabia", "genre": "Adventure, Biography, Drama", "year": 1962, "plot": "The story of T.E. Lawrence, the English officer who united rival Arab tribes to fight the Turks during World War I."},
    {"title": "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb", "genre": "Comedy", "year": 1964, "plot": "An insane general triggers a path to nuclear holocaust that a war room full of politicians and generals frantically try to stop."},
    {"title": "The Good, the Bad and the Ugly", "genre": "Western", "year": 1966, "plot": "A bounty hunter, a Mexican bandit, and a Union officer are all in pursuit of $200,000 worth of Confederate gold during the American Civil War."},
    {"title": "Once Upon a Time in the West", "genre": "Western", "year": 1968, "plot": "A mysterious stranger with a harmonica joins forces with a notorious desperado to protect a beautiful widow from a ruthless assassin working for the railroad."},
    {"title": "Alien", "genre": "Horror, Sci-Fi", "year": 1979, "plot": "The crew of a commercial space tug encounter a deadly alien lifeform after investigating a mysterious signal on a remote planet."},
    {"title": "Amadeus", "genre": "Biography, Drama, Music", "year": 1984, "plot": "The life, success and troubles of Wolfgang Amadeus Mozart as told by Antonio Salieri, a contemporary composer who was insanely jealous of Mozart's talent."},
    {"title": "Braveheart", "genre": "Biography, Drama, History", "year": 1995, "plot": "In 13th-century Scotland, William Wallace begins a revolt against King Edward I of England after the English murder his new bride."},
    {"title": "The Sixth Sense", "genre": "Drama, Mystery, Thriller", "year": 1999, "plot": "A boy who communicates with spirits who don't know they're dead seeks the help of a disheartened child psychologist."},
    {"title": "Donnie Darko", "genre": "Drama, Sci-Fi, Thriller", "year": 2001, "plot": "A troubled teenager is plagued by visions of a man in a large rabbit suit who manipulates him to commit a series of crimes."},
    {"title": "Eternal Sunshine of the Spotless Mind", "genre": "Drama, Romance, Sci-Fi", "year": 2004, "plot": "When their relationship turns sour, a couple undergoes a procedure to have each other erased from their memories."},
    {"title": "A Scanner Darkly", "genre": "Animation, Crime, Drama", "year": 2006, "plot": "In a dystopian future, a government agent goes undercover to infiltrate a drug distribution ring, but he begins to lose his own identity in the process."},
    {"title": "Moonlight", "genre": "Drama", "year": 2016, "plot": "A young man grapples with his identity and sexuality while experiencing the childhood, adolescence, and adulthood of his life."},
    {"title": "Get Out", "genre": "Horror, Mystery, Thriller", "year": 2017, "plot": "A young African-American man visits his white girlfriend's parents for the weekend, where his simmering uneasiness about the situation eventually boils over into a terrifying reality."},
    {"title": "Lady Bird", "genre": "Comedy, Drama", "year": 2017, "plot": "An adolescent girl navigates her last year of high school in Sacramento, California, and her complicated relationship with her mother."},
    {"title": "Three Billboards Outside Ebbing, Missouri", "genre": "Crime, Drama", "year": 2017, "plot": "A mother challenges local authorities to solve her daughter's murder when they fail to catch the culprit."},
    {"title": "The Shape of Water", "genre": "Adventure, Drama, Fantasy", "year": 2017, "plot": "At a top secret research facility in the 1960s, a lonely janitor forms a unique relationship with an amphibious creature that is being held in captivity."},
    {"title": "Joker", "genre": "Crime, Drama, Thriller", "year": 2019, "plot": "A mentally troubled comedian embarks on a downward spiral that leads to the creation of an iconic villain."},
    {"title": "1917", "genre": "Drama, War", "year": 2019, "plot": "Two young British soldiers during the First World War are given an impossible mission: deliver a message deep in enemy territory."},
    {"title": "Once Upon a Time... in Hollywood", "genre": "Comedy, Drama", "year": 2019, "plot": "A faded television actor and his stunt double strive to achieve success in the film industry during the final year of Hollywood's Golden Age in 1969 Los Angeles."},
    {"title": "Marriage Story", "genre": "Drama, Romance", "year": 2019, "plot": "A stage director and his actress wife struggle through a grueling coast-to-coast divorce that pushes them to their personal and creative extremes."},
    {"title": "Nomadland", "genre": "Drama", "year": 2020, "plot": "After losing everything in the Great Recession, a woman embarks on a journey through the American West as a modern-day nomad living in a van."},
    {"title": "Dune", "genre": "Action, Adventure, Drama", "year": 2021, "plot": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes burdened with a mysterious gift."},
    {"title": "CODA", "genre": "Drama, Music", "year": 2021, "plot": "As a CODA (Child of Deaf Adults), Ruby is the only hearing person in her family. When her family's fishing business is threatened, Ruby finds herself torn between pursuing her love of music and her fear of abandoning her parents."},
    {"title": "Everything Everywhere All at Once", "genre": "Action, Adventure, Comedy", "year": 2022, "plot": "A middle-aged Chinese immigrant is swept up in an insane adventure where she alone can save existence by exploring other universes and connecting with the lives she could have led."},
    {"title": "Oppenheimer", "genre": "Biography, Drama, History", "year": 2023, "plot": "The story of J. Robert Oppenheimer, the scientist who directed the Manhattan Project, leading to the development of the atomic bomb."}
]


    # DataFrame oluşturma ve CSV'ye kaydetme mantığı aynı kalır.
    df_movie = pd.DataFrame(movie_data_list)
    df_movie.to_csv(filename, index=False, encoding='utf-8')

# Not: Fonksiyonun geri kalanı (get_qa_chain) aynı kalacak.

# --- C. RAG PIPELINE KURULUMU ---
def get_qa_chain():
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmalıdır. Streamlit Cloud'da Secrets bölümünden ayarlayın.")

    # 1. Veri Hazırlığı
    create_and_save_data(FILE_NAME) # Uygulama her çalıştığında veriyi oluşturur.
    loader = CSVLoader(file_path=FILE_NAME, encoding="utf-8", csv_args={'delimiter': ','}, source_column="title")
    documents = loader.load()

    # 2. Embedding ve Vektör Veritabanı
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # ChromaDB (Oluştur ve kaydet)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME)
    # Streamlit Cloud'da diske yazmak yerine sadece bellekte tutuyoruz, çünkü disk kalıcı değildir.

    # 3. RAG Zincirini Kurma
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2) 
    
    prompt_template = """Sen, yalnızca sana sağlanan film veritabanından öneri yapabilen bir film öneri asistanısın.
Kullanıcıya nazikçe, akıcı bir dille cevap ver. Önerdiğin filmlerin özetini ve türünü açıklayarak önerini gerekçelendir.
Eğer verilen filmler (CONTEXT) kullanıcının sorusuna uygun değilse, kibarca "Üzgünüm, veri tabanımda bu kritere uygun bir film bulamadım." diye cevap ver.
Cevaplarında önerdiğin film adı, türü ve yılı mutlaka yer almalıdır.

CONTEXT (ChromaDB'den gelen en ilgili filmler):
{context}

Kullanıcının Sorusu: {question}

Cevabın:"""

    RAG_PROMPT_GEMINI = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": RAG_PROMPT_GEMINI}
    )
    return qa_chain

if __name__ == "__main__":
    # Konsol testi
    qa_chain = get_qa_chain()
    test_query = "Karmaşık kurgusu olan, bilim kurgu türünde bir film önerisi yap."
    print(f"TEST EDİLİYOR: {test_query}")
    result = qa_chain.invoke({"query": test_query})
    print("
MODEL CEVABI:")
    print(result["result"])
