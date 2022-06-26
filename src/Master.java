/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author sangeetdas
 */
public class Master {
    
    
    static int parseInt(String numStr){
        
        int num = 0;
        int n = numStr.length();
        boolean isNegative = false;
        int i = 0;
        if(numStr.charAt(0) == '-'){
            isNegative = true;
            i = 1;
        }
        
        for(; i < n; i++){
            
            num = num * 10 + (numStr.charAt(i) - '0');
        }
        return isNegative ? -1 * num : num;
    }
    
    
    static boolean isLetter(char ch){
        
        return (ch >= 'a' && ch <= 'z') 
                || (ch >= 'A' && ch <= 'Z');
        
    }
    
    static void reverseSpeacialString(String str){
        
        int n = str.length();
        char[] ch = str.toCharArray();
        
        int start = 0;
        int end = n - 1;
        
        while(end > start){
            
            while(start < end && !isLetter(str.charAt(start))){
                start++;
            }
            
            while(start < end && !isLetter(str.charAt(end))){
                end--;
            }
            
            char temp = ch[start];
            ch[start] = ch[end];
            ch[end] = temp;
            
            start++;
            end--;
        }
        
        //output
        System.out.println("Output: "+String.valueOf(ch));
        
    }
    
    
    public static void main(String[] args) {
        String numStr = "123";
        System.out.println(parseInt(numStr));
        numStr = "-123";
        System.out.println(parseInt(numStr));
        numStr = "0";
        System.out.println(parseInt(numStr));
        numStr = "1298034";
        System.out.println(parseInt(numStr));
        
        //second ques
        String str = "Sa@ng%&e!et";
        reverseSpeacialString(str);
        str = "@!&(&%";
        reverseSpeacialString(str);
        
        
        /*
        Game: 
        Cell[][] board
        Player current
        Player player1, player2
        List<Move> moves
        
        Gaame(player1, player2){
            innitGame();
        }
        
        private initGame(){
        
            borad[][] = new Cell[][] // 8 * 8
            //set white side pieces
            board[x][y - 1] = new Cell(x, y, PeiceType.ROOK, Color.WHITE)
            board[x][y] = new Cell(x, y, PeiceType.PAWNS, Color.WHITE)
            //set empty cells
            board[x][y] = new Cell(x, y, NULL, NULL)
            //set black side pieces
        
            current = player1.isWhite ? player1 : player2
        }
        
       makeMove(){
            
        //i/p = x, y, piece || x, y , dest
        
        //validate I/P(Cell @ x,y) == isValidPiece 
        
        PieceType currPiece = Cell(x,y)
        
        Move move = Move(srcPos[x1, y1], dest[x2, y2], current)
        
        destCell = board[x2][y2]
        
        destCell.setType = currPiece.getType
        destCell.setColor = currPiece.getColor
        
        currPiece.setType = Null
        currPiece.setColor = Null
        
        moves[] = move;
        
        current = current == player1 ? player2 : player2
        
        }
        
        */
        
        //board[][] 
        //Position{x, y}
        //Move(Position: src{x, y}, dest{x, y}, Player player)
        //Snapshot{ Game, }
        //List<Moves>
        
        
    }
}
