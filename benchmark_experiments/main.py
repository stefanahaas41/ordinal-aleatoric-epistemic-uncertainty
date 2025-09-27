from benchmark_experiments.abalone import abalone
from benchmark_experiments.automobile import automobile
from benchmark_experiments.autompg import autompg
from benchmark_experiments.balance import balance
from benchmark_experiments.breast import breast
from benchmark_experiments.cmc import cmc
from benchmark_experiments.era import era
from benchmark_experiments.esl import esl
from benchmark_experiments.eucalyptus import eucalyptus
from benchmark_experiments.grub_damage import grub_damage
from benchmark_experiments.heart import heart
from benchmark_experiments.housing import housing
from benchmark_experiments.lev import lev
from benchmark_experiments.machine import machine
from benchmark_experiments.new_thyroid import new_thyroid
from benchmark_experiments.obesity_level import obesity
from benchmark_experiments.pyrimidines import pyrimidines
from benchmark_experiments.red_wine import red_wine
from benchmark_experiments.stock import stock
from benchmark_experiments.swd import swd
from benchmark_experiments.tae import tae
from benchmark_experiments.triazines import triazines
from benchmark_experiments.white_wine import white_wine



def main():
    abalone.run()
    automobile.run()
    autompg.run()
    balance.run()
    breast.run()
    cmc.run()
    era.run()
    esl.run()
    eucalyptus.run()
    grub_damage.run()
    heart.run()
    housing.run()
    lev.run()
    machine.run()
    new_thyroid.run()
    obesity.run()
    pyrimidines.run()
    red_wine.run()
    stock.run()
    swd.run()
    tae.run()
    triazines.run()
    white_wine.run()


if __name__ == '__main__':
    main()
