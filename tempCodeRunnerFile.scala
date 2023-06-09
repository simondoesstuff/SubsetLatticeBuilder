def count(list: List[Int]): Int = {
    if (list.isEmpty) 0
    else if (list.head == 0) 1 + count(list.tail)
    else count(list.tail)
}


// println(count(List(1,4,0,0,15,1)))


def main() = {
    println("hi")
}