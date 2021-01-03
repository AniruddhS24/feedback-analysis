from models.featurescorer import *
from models.extractor import *
def main():
    fts = FeatureImportanceScorer('suppmodeel.pt')
    ext = HeuristicExtractor(fts)
    rats, rationalev = ext.contiguous_discretize(x=['This place is absolute garbage... Half of the tees are not available, including all the grass tees',
                                                    'I’m a professional makeup artist and I’ve been in search of brushes that are good quality but not super expensive and I came across these and I’m pleasantly surprised! I usually have multiple clients back to back so sometimes I don’t have time to clean my brushes so I needed some backups and now I find myself using these just as much as my Morphe and Real techniques ones (which I love). The bristles are super soft and they don’t shed at all. The face brushes are super dense and don’t soak up too much product. The eye brushes are soft and can blend and pack on shadow really well. I also didn’t notice any odd smells. I would definitely reccomend this!',
                                                    'I purchased the 6 piece set to this a couple years ago. I have other brands of brushes like elf studio professional and Morphe studio professional and that 6 piece set I bought is my fav. The bristles were so soft and dense and fluffy. So I Came back to purchase the 16 piece set. The bristles are stiffer and less dense. I\'ll not be purchasing this again. It was disappointing.'])
    print(rats)


if __name__ == '__main__':
    main()